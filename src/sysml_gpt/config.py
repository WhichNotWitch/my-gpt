""""超参数文件"""
from dataclasses import dataclass

@dataclass
class TrainConfig:
    input_path:str = "input.txt"
    checkpoint_path:str = "checkpoints/tiny_gpt.pt"
    best_checkpoint_path : str = "checkpoints/tiny_gpt_best.pt"

    batch_size:int =32
    block_size:int =64
    train_steps : int = 200
    eval_interval:int =100
    eval_iters:int=20
    learning_rate:float =3e-4

    n_embed:int =64
    n_layer:int=4
    num_heads:int=4
    dropout:float =0.2
    resume:bool = True
    resume_path:str = "checkpoints/tiny_gpt.pt"