import tempfile
from pathlib import Path
import unittest

import torch

from idle_rft.common import atomic_json, digest
from idle_rft.opd import (checkpoint_info, next_route, response_positions,
                          row_at, topk_reverse_divergence, validate_targets)


class DistillationTests(unittest.TestCase):
    def test_equal_distribution_has_zero_loss_and_gradient(self):
        logits = torch.tensor([[2., -1., 0., 1.]], requires_grad=True)
        teacher = logits.detach().log_softmax(-1)
        logq, ids = teacher.topk(2, -1)
        loss, pmass, qmass = topk_reverse_divergence(logits, logq, ids)
        loss.sum().backward()
        self.assertAlmostEqual(loss.item(), 0., places=6)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        torch.testing.assert_close(pmass, qmass)
        self.assertLess(qmass.item(), 1.)

    def test_full_support_matches_reverse_kl_and_gradient(self):
        logits = torch.tensor([[.2, .7, -.4]], requires_grad=True)
        teacher = torch.tensor([[-1., 2., .5]]).log_softmax(-1)
        logq, ids = teacher.topk(3, -1)
        loss, _, _ = topk_reverse_divergence(logits, logq, ids)
        expected = (logits.softmax(-1) * (logits.log_softmax(-1) - teacher)).sum()
        torch.testing.assert_close(loss.sum(), expected)
        actual_grad = torch.autograd.grad(loss.sum(), logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_descent_moves_toward_teacher_and_partial_loss_nonnegative(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        teacher = torch.tensor([[4., 0., -1., -2.]]).log_softmax(-1)
        logq, ids = teacher.topk(2, -1)
        loss, _, _ = topk_reverse_divergence(logits, logq, ids)
        loss.sum().backward()
        moved = logits.detach() - .1 * logits.grad
        next_loss, _, _ = topk_reverse_divergence(moved, logq, ids)
        self.assertGreater(loss.item(), 0.)
        self.assertLess(next_loss.item(), loss.item())
        self.assertGreater(moved.softmax(-1)[0, 0].item(), .25)

    def test_shift_targets_first_and_last_response_tokens(self):
        positions = response_positions(5, 3)
        self.assertEqual(list(range(8))[positions], [4, 5, 6])
        with self.assertRaises(ValueError):
            response_positions(0, 2)

    def test_route_quotas_and_reproducible_resumable_prompt_order(self):
        targets = {"bank": 12, "mbpp": 4}
        tokens = dict.fromkeys(targets, 0)
        routes = []
        while (route := next_route(tokens, targets)) is not None:
            tokens[route] += 1
            routes.append(route)
        self.assertEqual(tokens, targets)
        self.assertEqual(routes.count("mbpp"), 4)
        rows = [{"id": i} for i in range(7)]
        uninterrupted = [row_at(rows, i, 42) for i in range(20)]
        resumed = [row_at(rows, i, 42) for i in range(9, 20)]
        self.assertEqual(uninterrupted[9:], resumed)
        self.assertEqual(len({r["id"] for r in uninterrupted[:7]}), 7)

    def test_single_task_zero_quota_never_enters_disabled_route(self):
        targets = validate_targets({"bank": 12, "mbpp": 0})
        progress = {"bank": 0, "mbpp": 0}
        for _ in range(12):
            self.assertEqual(next_route(progress, targets), "bank")
            progress["bank"] += 1
        self.assertIsNone(next_route(progress, targets))
        for invalid in ({"bank": 0, "mbpp": 0}, {"bank": -1, "mbpp": 2},
                        {"bank": 1.5, "mbpp": 0}, {"bank": True, "mbpp": 0}):
            with self.assertRaises(ValueError):
                validate_targets(invalid)

    def test_checkpoint_rejects_incomplete_and_tampered_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "weights").write_text("test-only")
            atomic_json(root / "contract.json", {"base": "base", "response_tokens": 100})
            torch.save({"progress": {"tokens": 99}}, root / "state.pt")
            names = ["adapter/weights", "contract.json", "state.pt"]
            atomic_json(root / "files.json", {n: digest(root / n) for n in names})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                checkpoint_info(adapter, "base")
            torch.save({"progress": {"tokens": 100}}, root / "state.pt")
            atomic_json(root / "files.json", {n: digest(root / n) for n in names})
            contract, fingerprint = checkpoint_info(adapter, "base")
            self.assertEqual(contract["response_tokens"], 100)
            self.assertEqual(len(fingerprint), 64)
            with self.assertRaisesRegex(ValueError, "provenance"):
                checkpoint_info(adapter, "different-base")
            (adapter / "weights").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                checkpoint_info(adapter, "base")


if __name__ == "__main__":
    unittest.main()
