"""超参数文件"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    input_path: str = "input.txt"

    checkpoint_path: str = "checkpoints/tiny_gpt.pt"
    best_checkpoint_path: str = "checkpoints/tiny_gpt_best.pt"
    resume_path: str = "checkpoints/tiny_gpt.pt"

    log_path: str = "runs/train_log.csv"
    config_snapshot_path: str = "runs/config.json"
    run_dir: str | None = None

    batch_size: int = 32
    block_size: int = 64
    train_steps: int = 200
    eval_interval: int = 100
    eval_iters: int = 20
    learning_rate: float = 3e-4

    n_embed: int = 64
    n_layer: int = 4
    num_heads: int = 4
    dropout: float = 0.2

    resume: bool = True