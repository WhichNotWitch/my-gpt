当然，README 用中文更适合你后续复习和展示。你可以把 `README.md` 写成下面这个版本。

```markdown
# SysML GPT

这是一个面向 SysML v2 补全任务的小型字符级 GPT 项目。

本项目使用 PyTorch 从零实现一个 tiny GPT，并使用 uv 管理 Python 环境和依赖。项目目标不是直接得到工业级补全模型，而是以工程化方式逐步理解 GPT 的训练、生成、保存、恢复和测试流程。

input.txt的数据来源于官方 Systems-Modeling/SysML-v2-Release 仓库([https://github.com/Systems-Modeling/SysML-v2-Release.git])

## 功能特性

- 字符级 tokenizer
- 训练集 / 验证集切分
- 语言模型 mini-batch 构造
- Bigram baseline
- Tiny GPT 模型，包括：
  - token embedding
  - position embedding
  - masked multi-head self-attention
  - feedforward network
  - residual connection
  - layer normalization
  - dropout
- checkpoint 保存
- resume training，从已有 checkpoint 继续训练                       
- 支持 temperature 和 top-k sampling 的文本生成
- 使用 pytest 测试 tokenizer 和数据 batch 逻辑

## 项目结构

```text
src/sysml_gpt/
  config.py       # 训练配置
  tokenizer.py    # 字符级 tokenizer
  data.py         # 数据读取、切分和 batch 构造
  model.py        # Bigram 和 TinyGPT 模型
  train.py        # 训练入口
  generate.py     # 文本生成入口

tests/
  test_tokenizer.py
  test_data.py

input.txt         # SysML v2 训练材料
checkpoints/      # 模型 checkpoint
```

## 环境准备

同步依赖：

```powershell
uv sync
```

验证 PyTorch 是否可用：

```powershell
uv run python -c "import torch; print(torch.__version__)"
```

## 训练模型

从零开始训练：

```powershell
uv run python -m sysml_gpt.train --no-resume --train-steps 2000 --checkpoint-path checkpoints/tiny_gpt.pt
```

从已有 checkpoint 继续训练：

```powershell
uv run python -m sysml_gpt.train --resume --train-steps 500 --checkpoint-path checkpoints/tiny_gpt.pt
```

参数说明：

- `--no-resume`：不加载旧模型，从零开始训练。
- `--resume`：加载已有 checkpoint 继续训练。
- `--train-steps`：本次训练继续运行多少步。
- `--checkpoint-path`：checkpoint 保存或加载路径。

## 生成 SysML v2 文本

示例：

```powershell
uv run python -m sysml_gpt.generate --start "package " --max-new-tokens 300 --temperature 0.8 --top-k 5 --seed 42
```

参数说明：

- `--start`：生成时的起始文本。
- `--max-new-tokens`：最多生成多少个新字符。
- `--temperature`：控制随机性，越低越保守，越高越发散。
- `--top-k`：每一步只从概率最高的 k 个字符中采样。
- `--seed`：随机种子，用于复现实验结果。

常用生成设置：

- `temperature=0.6`：更保守，结构更稳定，但可能重复。
- `temperature=0.8`：比较平衡。
- `temperature=1.0`：更随机，更有变化，但更容易混乱。
- `top_k=5`：更稳定，但容易重复。
- `top_k=20`：更自由，但可能更乱。

## 运行测试

```powershell
uv run pytest
```

当前测试覆盖：

- tokenizer 编码和解码是否互逆
- vocab size 是否正确
- train/val split 是否正确
- batch 的 `x` 和 `y` shape 是否正确
- `y` 是否是 `x` 右移一位

## 当前实验结论

### Character-level baseline

字符级模型可以较快降低 loss，但生成时容易出现关键词拼写错误和重复字符，例如 `attttribute`、`Controlller`。

### Byte-level BPE tokenizer

Byte-BPE 使用 UTF-8 字节作为初始 token，并通过 BPE merge 学习常见 SysML 片段。当前较稳定的实验是：

```text
run: runs/byte_bpe_vocab500
tokenizer: byte-bpe
vocab_size: 500
block_size: 128
n_embed: 128
n_layer: 4
num_heads: 4
dropout: 0.3
```

## 当前限制

- 生成结果只是“风格相似”，不能保证语法正确。
- 还没有 SysML v2 grammar checker。
- 还没有面向 IDE 的补全接口。

## 后续方向

1. 扩充 SysML v2 训练数据。
2. 记录每次训练的 loss 曲线和配置。
3. 加入 SysML v2 语法检查或格式化。
4. 增加更完整的 CLI，例如 `sysml-gpt train` 和 `sysml-gpt complete`。
5. 尝试用更大的模型参数训练，例如更多层数、更大的 embedding。
6. 探索基于 prompt 的补全任务，而不是单纯续写。
```

写完后运行：

```powershell
uv run pytest
```

再验证 README 里的生成命令：

```powershell
uv run python -m sysml_gpt.generate --start "package " --max-new-tokens 100 --temperature 0.8 --top-k 5 --seed 42
```
