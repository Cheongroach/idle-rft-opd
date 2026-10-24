# Weekend RFT + OPD：最终实测

全部声明的 GPU 任务已经终止，完整 checkpoint、候选归档和文件 SHA256 已核对。OPD 接在完成的 MBPP SFT 后，bank expert 提供新增任务监督，MBPP teacher 保持原任务。

## 原任务：候选筛选与后训练

Qwen3.5-4B、rank16。bank 原模型生成 24,576 候选，18,939 通过测试；A/B 各 4,235 题，A 随机选题、B 优先低原模型成功率。18 个配置各完成 1,000,000 监督 token，以全部开发集选择公共 LR 2e-4。

| bank 组 | seed42 | seed43 | seed44 | 均值 | seed 标准差 |
|---|---:|---:|---:|---:|---:|
| A | 97.34% | 97.42% | 98.36% | 97.71% | 0.57 pp |
| B | 98.59% | 98.28% | 98.91% | 98.59% | 0.31 pp |

bank blind base：85.08%，每模型 1,280 题。逐种子配对 CI 及 A/B 对比见 bank-final.md。

MBPP blind：base 95/170、SFT A 105/170，+5.88 pp，配对 11 胜/1 负，95% CI [+2.34,+9.94] pp；仅 seed42。

## OPD 接入与对照

Teacher 资格：bank dev student 664/768，teacher 760/768；+12.50 pp，95% CI [10.16,14.97] pp，配对 98 胜/2 负。

三种子同步 OPD 对比三种子 answer-only replay。每配置严格 16,384 响应 token，bank 12,288 / MBPP 4,096，LR 5e-5、四响应累积、token-based LR decay。六组均有有限 loss/梯度、实际参数变化、非空 Adam 状态和完整 checkpoint。

teacher top64 的目标为 `sum[p*(log(p)-log(q))-p+q]`；p/q 保留全词表归一化，这是有遗漏 tail 的近似。student 以 T=1、top-p=1、top-k=0 新鲜采样前缀；teacher 独立冻结。

| 任务 | warm | bank expert | OPD 均值 ± seed std | replay 均值 ± seed std |
|---|---:|---:|---:|---:|
| bank | 85.39% | 98.59% | 94.66% ± 2.58 pp | 90.13% ± 2.50 pp |
| mbpp | 61.76% | 58.82% | 61.37% ± 0.90 pp | 58.82% ± 0.59 pp |

逐任务判读：
- bank：OPD 相对 warm 的均值变化 +9.27 pp；相对 replay +4.53 pp；距对应任务专家 +3.93 pp。三组同 seed 配对 CI 中，2 组完全为正、0 组完全为负、1 组包含零。
- mbpp：OPD 相对 warm 的均值变化 -0.39 pp；相对 replay +2.55 pp；距对应任务专家 +0.39 pp。三组同 seed 配对 CI 中，0 组完全为正、0 组完全为负、3 组包含零。

所有 seed 均进入盲测，没有根据 blind 选择 seed 或追加训练。逐种子相对 warm、相对任务专家及 OPD/replay 配对比较见 opd-final.md/JSON。
上述均值差是描述性结果，逐 seed 配对 CI 不能替代跨训练种子的总体显著性检验。
MBPP 相对 warm 的三个 OPD 配对 CI 均包含零，下界最低约 −4.52 pp；没有显著下降不等于证明非劣，本轮没有部署非劣或线上 fallback 验收。

## 计算成本

全部 42 个任务的完整 Pod 区间合计 172.98 H20 卡时；峰值同时分配 64 卡。保守预留 721.00 卡时，上限 3000。

- opd 三种子进程合计 0.8213 卡时，平均响应 token/s 16.64；峰值显存 17.12 GiB。
- replay 三种子进程合计 0.1515 卡时，平均响应 token/s 90.38；峰值显存 13.03 GiB。

进程计时包括加载、rollout、teacher 评分、更新和保存；共享 Pod 的准备、空等与导出另计。等响应 token 不等计算预算，未分离 teacher prefill 的阶段耗时。以上为资源时间观察，平台账单未验证。

## 验证范围与可讲结论

这是同规模、同 tokenizer 的两个代码任务集融合实验。MBPP 是公开题，存在历史污染；bank 划分共享操作族，验证新规格泛化。A/B 等监督 token，但答案长度、优化步数与选题组成仍可能不同。
OPD 与 replay 的轨迹和监督形式不同，不能把方法差异单独归因于 on-policy。本轮没有单任务 OPD 消融，无法把蒸馏迁移缺口与混合训练干扰分开。bank teacher 训练为 100 万监督 token，融合每配置仅 1.6384 万响应 token；本轮是冻结的小预算接入实验，没有根据 blind 追加预算。

未执行纯 base 预训练、多域共同 SFT、奖励式 RL、异步 rollout、sampled-token PG、全词表 KL 或异源 tokenizer 蒸馏。OPD 冷进程 GPU 恢复尚未单独验证；代码复用了已验证的恢复实现，CPU 测试覆盖窄逻辑。

代码：[main](https://github.com/Cheongroach/idle-rft-opd/tree/main)。手稿、真实对话摘录、脱敏日志及结果：[research](https://github.com/Cheongroach/idle-rft-opd/tree/research)。
