# OPD 补强：预算曲线、单任务对照与新留出集

全部36组训练、38个开发集模型、20个新留出模型及38个固定前缀诊断已完成。开发集选择响应token预算 524,288；保持性选型门槛通过：True。
完整开发集1,154题（bank 1,024、MBPP 130），合并原开发集与新增256/80题；新留出集2,484题（bank 2,048、MBPP 436）。

## 开发集预算曲线

| 预算 | OPD bank | replay bank | OPD retention | replay retention |
|---|---:|---:|---:|---:|
| 65,536 | 97.72% | 95.48% | 59.74% | 59.49% |
| 262,144 | 98.89% | 95.54% | 58.46% | 57.95% |
| 524,288 | 99.02% | 97.14% | 59.49% | 57.18% |

最大两档bank dev均值变化 +0.13pp；冻结的平台期判据满足：True。独立预算训练各有不同LR horizon，不能拼成同一优化轨迹或证明全域收敛。

## 新留出集三种子汇总

| 方法 | bank mean ± seed std | retention mean ± seed std |
|---|---:|---:|
| opd | 98.44% ± 0.13pp | 46.64% ± 0.35pp |
| replay | 96.91% ± 0.24pp | 45.18% ± 0.23pp |
| bank-only | 98.45% ± 0.07pp | 46.02% ± 0.48pp |
| balanced | 98.16% ± 0.15pp | 46.41% ± 0.35pp |
| bank-only-balanced | 98.34% ± 0.10pp | 46.02% ± 0.13pp |
| long-prefix | 96.88% ± 0.70pp | 46.56% ± 0.61pp |

warm：bank 1792/2048；retention 202/436。

expert：bank 2019/2048；retention 197/436。

## 新留出集

| 方法 | seed | bank | retention | retention vs warm CI | 非劣(-2pp) |
|---|---:|---:|---:|---|---|
| opd-524288-s42 | 42 | 98.39% | 46.33% | [-1.37,+1.37]pp | True |
| opd-524288-s43 | 43 | 98.34% | 46.56% | [-1.58,+2.05]pp | True |
| opd-524288-s44 | 44 | 98.58% | 47.02% | [-0.46,+1.85]pp | True |
| replay-524288-s42 | 42 | 96.63% | 44.95% | [-3.67,+0.90]pp | False |
| replay-524288-s43 | 43 | 97.07% | 45.18% | [-3.46,+1.14]pp | False |
| replay-524288-s44 | 44 | 97.02% | 45.41% | [-3.23,+1.38]pp | False |
| bank-only-524288-s42 | 42 | 98.39% | 45.87% | [-2.30,+1.38]pp | False |
| bank-only-524288-s43 | 43 | 98.44% | 45.64% | [-2.74,+1.37]pp | False |
| bank-only-524288-s44 | 44 | 98.54% | 46.56% | [-1.84,+2.29]pp | True |
| balanced-s42 | 42 | 98.00% | 46.10% | [-1.60,+1.15]pp | True |
| balanced-s43 | 43 | 98.19% | 46.79% | [-0.69,+1.83]pp | True |
| balanced-s44 | 44 | 98.29% | 46.33% | [-1.38,+1.38]pp | True |
| bank-only-balanced-s42 | 42 | 98.34% | 46.10% | [-2.08,+1.61]pp | False |
| bank-only-balanced-s43 | 43 | 98.24% | 46.10% | [-2.26,+1.83]pp | False |
| bank-only-balanced-s44 | 44 | 98.44% | 45.87% | [-2.29,+1.37]pp | False |
| long-prefix-s42 | 42 | 97.07% | 47.25% | [-0.68,+2.53]pp | True |
| long-prefix-s43 | 43 | 97.46% | 46.10% | [-1.84,+1.38]pp | True |
| long-prefix-s44 | 44 | 96.09% | 46.33% | [-1.61,+1.60]pp | True |

## 固定前缀分布诊断

在相同64个bank和16个MBPP前缀上评分，adapter/teacher/前缀哈希及逐题token覆盖均已校验。下表为逐token加权距离再跨seed取均值；单个控制模型没有跨seed平均。

| 方法 | bank top64 distance | MBPP top64 distance |
|---|---:|---:|
| warm | 0.010148 | 0.000245 |
| expert | 0.000000 | 0.008344 |
| opd | 0.000352 | 0.001010 |
| replay | 0.002245 | 0.035134 |
| bank-only | 0.000358 | 0.004965 |
| balanced | 0.000445 | 0.000811 |
| bank-only-balanced | 0.000491 | 0.004751 |
| long-prefix | 0.001613 | 0.001029 |

该距离是保留全词表概率归一化的截断广义reverse divergence，不是完整词表KL。距离下降只说明这些固定条件前缀上的分布接近，不能代替程序成功率、证明完整收敛或teacher纠错续写能力。

实测发现PEFT 0.18.1加载第二个MBPP teacher adapter时先经BF16舍入，再转为FP32。源checkpoint相同，但运行时LoRA权重与warm student不逐元素相同。最大参数差0.00006104；在一个236-token固定前缀上，top64距离0.00024391。临时teacher直接回填源FP32权重后，全部LoRA参数相同，该前缀输出差异与距离均为0。官方36组实验保留冻结的加载行为；不将warm MBPP非零距离解读为学习误差。该附加诊断使用已计入Pod成本的完成进程空闲GPU；未保存adapter。明细见teacher-load-precision-audit.json。

同bank token配额的mixed/single配对比较、专家差距和OPD/replay逐seed区间见 `final-analysis.json`。固定前缀距离见 `fixed-prefix-diagnostics.json`。三种子标准差不是置信区间，没有显著下降不代表非劣通过。

## 成本和边界

开发集选定预算下，三个OPD进程合计 22.74 H20卡时，replay合计 3.19 卡时，比值 7.13x。进程时间包含加载、rollout、teacher评分、更新和保存；teacher评分未独立计时。恢复配置同时计入原中断进程；未记录正常退出时间的原进程以Pod停止时间给出上界，逐配置来源和上界标识见update-metrics.json。同节点较短进程结束后等待其他进程的空等计入Pod资源时间。逐模型输出截断与其他失败计数见final_suite_summaries。

原实验及补强共 79 个任务，完整Pod区间合计 521.34 H20卡时；保守预留 1853.10/3000卡时；峰值 64 卡，活跃GPU为0。不是平台账单。
新留出题与本项目旧题去重，不证明模型预训练无污染；规格题仍共享操作族。未执行奖励RL、异步rollout、异词表或完整词表loss消融。
