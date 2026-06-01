# SysML GPT

这是一个面向 SysML v2 文本补全的小型 GPT 项目。项目使用 PyTorch 从零实现 tiny GPT，并使用 `uv` 管理 Python 环境、依赖、训练、生成和测试流程。

当前项目目标不是直接得到工业级 SysML v2 补全器，而是以工程化方式逐步理解并实现：

- 语言模型数据构造
- tokenizer
- Transformer / GPT 模型
- checkpoint 保存与恢复
- CUDA 训练
- 固定 prompt 评估
- 采样策略调试

训练数据来自 SysML v2 官方示例仓库：

[Systems-Modeling/SysML-v2-Release](https://github.com/Systems-Modeling/SysML-v2-Release.git)

## 当前状态

项目已经完成两条主线：

1. 字符级 GPT baseline
2. Byte-level BPE tokenizer + tiny GPT

当前更推荐继续使用 Byte-BPE 实验线，因为它生成 SysML 关键词和局部结构时更稳定，重复字符和拼写漂移比字符级模型少。

当前较稳定的实验配置：

```text
run: runs/byte_bpe_vocab500
tokenizer: byte-bpe
vocab_size: 500
block_size: 128
batch_size: 16
n_embed: 128
n_layer: 4
num_heads: 4
dropout: 0.3
sampling: temperature=0.7, top_k=50, top_p=0.9
```

## 项目结构

```text
src/sysml_gpt/
  config.py         # 默认训练配置
  tokenizer.py      # CharTokenizer、CharBPETokenizer、ByteBPETokenizer
  data.py           # 文本读取、train/val token split、batch 构造
  model.py          # Bigram baseline 和 TinyGPT 模型
  train.py          # 训练入口，支持 resume、run-dir、checkpoint
  generate.py       # 单 prompt 生成入口
  eval_generate.py  # 固定 prompt 批量生成评估
  prepare_data.py   # 合并 data/raw 中的 SysML 文件

tests/
  test_tokenizer.py
  test_data.py

data/
  raw/              # 原始 .sysml/.txt/.kerml 文件
  processed/        # prepare_data.py 生成的训练文本

runs/               # 每次实验的 config、loss log、last/best checkpoint
checkpoints/        # 手动保存的重要模型
eval_prompts.txt    # 固定评估 prompt
```

## 环境准备

同步依赖：

```powershell
uv sync
```

本项目当前在 `pyproject.toml` 中固定使用 CUDA 12.1 版 PyTorch：

```text
torch==2.5.1+cu121
```

验证 PyTorch 和 CUDA：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

期望输出类似：

```text
2.5.1+cu121
12.1
True
NVIDIA GeForce RTX 2060
```

如果 `torch.cuda.is_available()` 是 `False`，说明当前环境没有使用 CUDA 版 PyTorch。

## 数据准备

把 SysML v2 示例文件放在：

```text
data/raw/
```

合并为训练语料：

```powershell
uv run python -m sysml_gpt.prepare_data --input-dir data/raw --output-path data/processed/train.txt
```

当前 `prepare_data.py` 会将 `data/raw` 下的 `.txt`、`.sysml`、`.kerml` 文件按文件名排序后合并，并在文件之间插入：

```text
<|endoftext|>
```

注意：当前项目暂时没有做文件级 train/val 切分。训练时仍然是在 token 序列上连续切分 train/val。

## 训练字符级 baseline

字符级 tokenizer 是最简单的 baseline：

```powershell
uv run python -m sysml_gpt.train --no-resume --train-steps 3000 --run-dir runs/char_baseline --input-path data/processed/train.txt --tokenizer char --block-size 128 --batch-size 16 --n-embed 128 --n-layer 4 --num-heads 4 --dropout 0.2
```

字符级模型通常 loss 下降较快，但生成时容易出现：

- 关键词拼写错误
- 重复字符
- 标识符漂移

例如：

```text
attttribute
Controlller
```

## 训练 Byte-BPE 模型

Byte-BPE 从 UTF-8 bytes 开始训练 merge 规则，更接近 GPT 系 tokenizer。它可以编码任意 Unicode 文本，也能把高频 SysML 片段合并成更稳定的 token。

推荐从 `vocab_size=500` 开始：

```powershell
uv run python -m sysml_gpt.train --no-resume --train-steps 3000 --run-dir runs/byte_bpe_vocab500 --input-path data/processed/train.txt --tokenizer byte-bpe --vocab-size 500 --block-size 128 --batch-size 16 --n-embed 128 --n-layer 4 --num-heads 4 --dropout 0.3
```

继续训练：

```powershell
uv run python -m sysml_gpt.train --resume --train-steps 3000 --run-dir runs/byte_bpe_vocab500 --input-path data/processed/train.txt
```

`--run-dir` 会把实验产物组织到同一个目录：

```text
runs/byte_bpe_vocab500/
  config.json
  train_log.csv
  last.pt
  best.pt
```

其中：

- `last.pt`：最后一步 checkpoint，主要用于 resume
- `best.pt`：验证集 loss 最低的 checkpoint，主要用于生成和评估

## 生成文本

单 prompt 生成：

```powershell
uv run python -m sysml_gpt.generate --checkpoint runs/byte_bpe_vocab500/best.pt --start "package " --max-new-tokens 120 --temperature 0.7 --top-k 50 --top-p 0.9 --seed 42
```

当前 Byte-BPE 推荐采样参数：

```text
temperature = 0.7
top_k = 50
top_p = 0.9
```

参数说明：

- `--start`：补全起始文本
- `--max-new-tokens`：最多生成多少个新 token。Byte-BPE 下一个 token 可能对应多个字符
- `--temperature`：控制随机性，越低越保守
- `--top-k`：只从概率最高的 k 个 token 中采样
- `--top-p`：只从累计概率达到 p 的候选集合中采样
- `--seed`：随机种子，用于复现实验

## 固定 prompt 评估

`eval_prompts.txt` 中可以放固定评估 prompt，例如：

```text
---
package 
---
part def 
---
attribute 
---
action def 
---
state def 
---
requirement def 
```

批量生成评估：

```powershell
uv run python -m sysml_gpt.eval_generate --checkpoint runs/byte_bpe_vocab500/best.pt --prompts eval_prompts.txt --output runs/byte_bpe_vocab500/eval_samples.txt --max-new-tokens 120 --temperature 0.7 --top-k 50 --top-p 0.9 --seed 42
```

查看结果：

```powershell
Get-Content runs\byte_bpe_vocab500\eval_samples.txt
```

固定 prompt 评估比单看 loss 更重要，因为字符级模型和 Byte-BPE 模型的 token 粒度不同，loss 数值不能直接横向比较。

## 测试

运行测试：

```powershell
uv run pytest
```

当前测试覆盖：

- tokenizer encode/decode 往返
- vocab size 基本逻辑
- train/val split
- batch shape
- `y` 是否是 `x` 右移一位

## 当前实验结论

### 字符级 baseline

优点：

- 实现简单
- 训练稳定
- loss 数值较容易下降

缺点：

- 逐字符生成，容易拼错关键词
- 容易出现重复字符
- 对 SysML 的结构单位没有显式建模

### Byte-level BPE

优点：

- 使用 UTF-8 bytes 作为初始 token，可以覆盖任意文本
- 高频 SysML 片段会被合并成更稳定的 token
- 生成关键词和局部结构更稳定
- 重复字符问题比字符级模型少

当前观察：

- `vocab_size=500` 比 `vocab_size=1000` 更稳
- Byte-BPE 的 loss 数值不能和字符级 loss 直接比较
- `temperature=0.7, top_k=50, top_p=0.9` 的生成结果较平衡

## 当前限制

- 当前生成仍然只是“SysML 风格相似”，不能保证语法正确
- 还没有 SysML v2 grammar checker
- 还没有文件级 train/val holdout
- 还没有 IDE 或 LSP 补全接口
- 当前模型规模较小，长距离结构和括号闭合仍不稳定

## 后续方向

建议后续按这个顺序推进：

1. 稳定 Byte-BPE 主线，继续比较 `vocab_size=500/800/1000`
2. 增加文件级 train/val 切分，让验证集更可信
3. 加入 SysML v2 语法检查或格式化
4. 保存更多实验元数据，例如最佳 step、生成样例、训练耗时
5. 做一个 `complete` 命令，专门服务 SysML prompt 补全
6. 尝试更大的模型，但优先保证数据和 tokenizer 质量

## 常用命令速查

准备数据：

```powershell
uv run python -m sysml_gpt.prepare_data --input-dir data/raw --output-path data/processed/train.txt
```

训练 Byte-BPE：

```powershell
uv run python -m sysml_gpt.train --no-resume --train-steps 3000 --run-dir runs/byte_bpe_vocab500 --input-path data/processed/train.txt --tokenizer byte-bpe --vocab-size 500 --block-size 128 --batch-size 16 --n-embed 128 --n-layer 4 --num-heads 4 --dropout 0.3
```

继续训练：

```powershell
uv run python -m sysml_gpt.train --resume --train-steps 3000 --run-dir runs/byte_bpe_vocab500 --input-path data/processed/train.txt
```

生成：

```powershell
uv run python -m sysml_gpt.generate --checkpoint runs/byte_bpe_vocab500/best.pt --start "package " --max-new-tokens 120 --temperature 0.7 --top-k 50 --top-p 0.9 --seed 42
```

批量评估：

```powershell
uv run python -m sysml_gpt.eval_generate --checkpoint runs/byte_bpe_vocab500/best.pt --prompts eval_prompts.txt --output runs/byte_bpe_vocab500/eval_samples.txt --max-new-tokens 120 --temperature 0.7 --top-k 50 --top-p 0.9 --seed 42
```

测试：

```powershell
uv run pytest
```
