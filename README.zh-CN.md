# 洄 · 可逆训练研究实现

**研究预发布版本：0.1.0rc1 · Apache-2.0。**

洄是一个面向显存受限因果语言模型训练的研究实现。它研究的问题是：在可逆残差块中保存 INT8 逆残差校正，能不能减少激活存储，同时让反向计算保持足够的数值保真。

同一个小模型可以切换三种后端：

- `exact`：普通 autograd，用作数值参考；
- `checkpoint`：完整激活检查点，用作实际资源基线；
- `n2`：可逆块使用 INT8 逆残差校正，普通块使用检查点。

## 洄可以用来做什么

洄适合希望下面这些问题的人：

- 观察有限精度重建怎样影响梯度和优化器更新；
- 在同一模型、同一配方下比较可逆重建和激活检查点；
- 设计显存受限的因果语言模型训练后端；
- 复现当前公开的 CPU 检查，并阅读已经审计的历史 GPU 结果。

它更像一个研究工具和后端原型，方便继续添加新的重建、量化或检查点策略。它不是现成的 Llama/Qwen 权重转换器，也不是生产训练框架；现有证据也不支持把 N2 说成在所有场景下都更快或更省显存。

## 现有证据怎么看

完整数据见 [RESULTS.md](RESULTS.md)、[主实验汇总](paper/data/audited-results.json) 和 [A800 资源对照](paper/data/a800-results.json)。

| 问题 | 当前证据 |
|---|---|
| 残差校正能否改善重建保真？ | Phase N 筛选中，N2 在 36/36 个检查点上通过已登记的直接保真门槛；未校正逆运算通过 24/36。 |
| 短程训练是否接近参考路径？ | 4 组 13.46 亿参数、5000 万 token 的配对训练满足内部工程容差，平均验证损失差为 −0.003759；4 个 seed 不能证明统计非劣。 |
| N2 是否胜过完整检查点？ | 在匹配的 A800 探针中没有。序列长度 4096 时，N2 吞吐约为检查点基线的 0.717 倍，allocated 显存约为 1.003 倍。 |
| 正向的显存结果是什么？ | 在归档的 Phase D 对照中，N2 相对保存激活的精确反向参考路径，allocated 显存约下降 43.2%。这个比较对象不是完整检查点。 |

上面的数字来自不同历史协议，不能合并成一个适用于所有硬件和模型的总评；当前 CPU 示例也不会复现历史 GPU 训练。

## 可能的使用场景

- **方法研究：** 比较可逆重建、激活检查点和激活精度策略；
- **显存受限实验：** 在固定显存预算下，探索更大的 batch 或更长的序列；
- **数值问题排查：** 分开检查梯度、AdamW 更新和保存恢复，不只看最终 loss；
- **教学和原型：** 先用小型 CPU 示例理解训练链路，再接入更大的受控实验。

## 当前边界

- 公开包没有重新运行历史 GPU 实验；
- CPU 示例使用小模型和随机 token，只检查链路和确定性恢复，不证明语言质量或 GPU 显存收益；
- N2 训练目前要求可逆块内所有参数保持可训练；
- 通用预训练权重转换和 LoRA/PEFT 不在本候选范围内；
- 技术报告是工作稿，不是同行评审论文。

## 安装与体验

本候选要求 Python 3.11+、PyTorch 2.8+。本地功能验证使用 Python 3.11、PyTorch 2.8.0、macOS CPU；不代表原有 CUDA 冻结环境。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.8.0' 'numpy==2.2.6'
python -m pip install --no-deps .
python -m hui.demo --backend exact --steps 5
python -m hui.demo --backend checkpoint --steps 5
python -m hui.demo --backend n2 --steps 5
```

示例使用 CPU/FP32 和随机合成 token，不下载语料、不调用 GPU、不启动后台服务。输出 JSON 包括 loss、步数、参数量及环境。它验证训练链路，不用于证明语言能力、显存收益或训练收敛。

保存和恢复：

```bash
python -m hui.demo --backend n2 --steps 2 --save partial.pt
python -m hui.demo --backend n2 --steps 5 --resume partial.pt --save resumed.pt
```

`--steps 5` 表示总目标为 5 步。已有文件不会被覆盖。恢复要求同一后端、seed 和小模型配置，并恢复模型、AdamW、数据生成器状态和 loss 历史。本入口仅读取该示例自身生成的检查点。

## 三种后端

| 后端 | 行为 | 用途 |
|---|---|---|
| `exact` | 普通精确反向 | 数值参考，API 默认 |
| `checkpoint` | 每层使用完整激活检查点 | 同模型强资源基线 |
| `n2` | 可逆层使用 INT8 逆误差校正，标准层使用检查点 | 可选研究后端 |

历史匹配实验中，完整检查点比 N2 更快，allocated 显存略低；这里不把 N2 包装为最佳默认。N2 的已有价值是指定配置下的数值保真与约 5000 万 token 训练中通过工程质量容差的证据；四个种子尚未建立统计非劣性。详见 [证据与边界](docs/EVIDENCE.md)。

```python
import torch
from hui import build_model, tiny_config

model = build_model(tiny_config(), "n2")
tokens = torch.randint(0, 64, (2, 16))
logits, loss = model(tokens, labels=tokens)
loss.backward()
```

先使用 `tiny_config()`；直接使用历史 `ExperimentConfig()` 默认值会构建约十亿参数规模的模型。已有 Llama/Qwen 权重的可逆转换不属于本候选功能。

N2 训练目前要求可逆块内全部参数保持可训练；若冻结其中的参数，入口会在前向前明确拒绝。部分冻结请使用 `exact` 或 `checkpoint`。冻结参数的普通推理不受影响；本 N2 候选不支持 LoRA/PEFT。

## 验证与发布边界

```bash
python -m unittest discover -s tests -v
python tools/verify_sources.py
```

原始快照中的三个核心文件仅调整包内导入，保留上游哈希与候选哈希。测试覆盖前向一致性、FP32/BF16 梯度、INT8 校正实际启用、有限更新和保存恢复；CPU 测试不替代原 1.3B/50M 实验。

没有复制权重、语料、凭据、部署配置或历史 Git。项目采用 Apache-2.0，代码来源说明和验证范围已随本版记录，见 [发布状态](RELEASE_STATUS.md)。

研究预发布版本：`0.1.0rc1`；集体署名：**WKK 实验室（WKK Lab）**。参见 [引用信息](CITATION.cff)、[第三方归属说明](THIRD_PARTY_NOTICES.md)、[发布说明](RELEASE_NOTES.md) 与 [源码来源](docs/SOURCE_PROVENANCE.md)。项目许可见 [LICENSE](LICENSE)，归属说明见 [NOTICE](NOTICE)；本版是研究快照，不代表生产训练框架或历史 GPU 实验的完整复现。

技术报告：[英文 PDF](paper/build/manuscript.pdf) · [中文全文](paper/中文完整译稿.md)；均为工作稿，未声称经过同行评审。[版本与下载](https://github.com/Hldao/hui-reversible-research/releases/tag/v0.1.0-rc1)。
