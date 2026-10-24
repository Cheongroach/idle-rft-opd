"""Local cached Qwen3.5 text model. No download or trust_remote_code path."""
from contextlib import contextmanager
import random

from .common import digest, read_json

TARGETS = ["q_proj", "v_proj", "in_proj_qkv", "in_proj_z"]
SYSTEM = ("Write a correct Python solution. Use the required function names and signatures. "
          "Return only Python code, without Markdown or explanation. "
          "Use the Python standard library only.")


def render(tokenizer, row):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": row["prompt"] + "\nRequired interface:\n"
          + "\n".join(row["interface"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


@contextmanager
def puretorch(modeling):
    names = ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
             "fused_recurrent_gated_delta_rule", "FusedRMSNormGated", "is_fast_path_available")
    previous = {n: getattr(modeling, n) for n in names}
    try:
        for name in names:
            setattr(modeling, name, False if name == "is_fast_path_available" else None)
        yield
    finally:
        for name, value in previous.items():
            setattr(modeling, name, value)


def check_assets(model_path, manifest_path):
    from pathlib import Path
    root = Path(model_path)
    manifest = read_json(manifest_path)
    for name, info in manifest["files"].items():
        local = (root / name).resolve()
        if not local.is_relative_to(root.resolve()):
            raise ValueError("Unsafe asset name")
        if local.stat().st_size != info["bytes"] or digest(local) != info["sha256"]:
            raise ValueError(f"Model checksum mismatch: {name}")
    return digest(manifest_path)


def load(model_path, manifest_path, rank=None, seed=42):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    if not torch.cuda.is_available():
        raise RuntimeError("Real-model execution requires one idle CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    fingerprint = check_assets(model_path, manifest_path)
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = Qwen3_5TextConfig.from_pretrained(model_path, local_files_only=True)
    config.attention_dropout = 0.0
    with puretorch(modeling):
        base, info = Qwen3_5ForCausalLM.from_pretrained(
            model_path, config=config, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Non-exact text model load: {info}")
    for layer in base.model.layers:
        if layer.layer_type == "linear_attention":
            block = layer.linear_attn
            block.causal_conv1d_fn = None
            block.causal_conv1d_update = modeling.torch_causal_conv1d_update
            block.chunk_gated_delta_rule = modeling.torch_chunk_gated_delta_rule
            block.recurrent_gated_delta_rule = modeling.torch_recurrent_gated_delta_rule
    model = attach(base, rank, seed) if rank is not None else base
    model.to("cuda")
    return model, tokenizer, fingerprint


def attach(base, rank, seed):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2,
            target_modules=TARGETS, lora_dropout=0., bias="none"))
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not parameters or any(".lora_" not in n for n, _ in parameters):
        raise RuntimeError("Only LoRA parameters may be trainable")
    for _, parameter in parameters:
        parameter.data = parameter.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model
