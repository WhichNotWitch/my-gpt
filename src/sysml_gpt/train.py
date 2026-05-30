"""训练"""


from pathlib import Path

import torch
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from sysml_gpt.data import get_batch,load_text,train_val_split
from sysml_gpt.model import TinyGPTLanguageModel
from sysml_gpt.tokenizer import CharTokenizer
from sysml_gpt.config import TrainConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--best-checkpoint-path",default=None)
    parser.add_argument("--log-path",default=None)
    parser.add_argument("--config-snapshot-path", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--n-embed", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    return parser.parse_args()

def apply_args(config: TrainConfig, args):
    if args.train_steps is not None:
        config.train_steps = args.train_steps

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    if args.block_size is not None:
        config.block_size = args.block_size

    if args.checkpoint_path is not None:
        config.checkpoint_path = args.checkpoint_path
        config.resume_path = args.checkpoint_path

    if args.best_checkpoint_path is not None:
        config.best_checkpoint_path = args.best_checkpoint_path

    if args.log_path is not None:
        config.log_path = args.log_path

    if args.config_snapshot_path is not None:
        config.config_snapshot_path = args.config_snapshot_path

    if args.resume:
        config.resume = True

    if args.no_resume:
        config.resume = False

    if args.run_dir is not None:
        config.run_dir = args.run_dir
        config.checkpoint_path = str(Path(args.run_dir) / "last.pt")
        config.best_checkpoint_path = str(Path(args.run_dir) / "best.pt")
        config.log_path = str(Path(args.run_dir) / "train_log.csv")
        config.config_snapshot_path = str(Path(args.run_dir) / "config.json")
        config.resume_path = config.checkpoint_path

    if args.input_path is not None:
        config.input_path = args.input_path

    if args.n_embed is not None:
        config.n_embed = args.n_embed

    if args.n_layer is not None:
        config.n_layer = args.n_layer

    if args.num_heads is not None:
        config.num_heads = args.num_heads

    if args.dropout is not None:
        config.dropout = args.dropout
    return config

def save_config_snapshot(config:TrainConfig):
    path = Path(config.config_snapshot_path)
    path.parent.mkdir(exist_ok=True)

    with path.open("w",encoding="utf-8") as f:
        json.dump(asdict(config),f,indent=2)


@torch.no_grad()
def estimate_loss(model,train_data,val_data,config):
    out ={}
    model.eval()

    for split,data in [("train",train_data),("val",val_data)]:
        losses = torch.zeros(20)

        for k in range(20):
            x,y = get_batch(data,batch_size=config.batch_size,block_size=config.block_size)
            x = x.to(device)
            y = y.to(device)

            _,loss = model(x,y)
            # loss是一个tensor，里面存了计算图等其他信息，用.item()仅仅取出数值
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out

def save_checkpoint(
        path,
        model,
        optimizer,
        step,
        tokenizer,
        config,
        best_val_loss,
):
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint ={
            "model_state_dict":model.state_dict(),
            "vocab_size":tokenizer.vocab_size,
            "stoi":tokenizer.stoi,
            "itos":tokenizer.itos,
            "block_size":config.block_size,
            "n_embed":config.n_embed,
            "n_layer": config.n_layer,
            "num_heads": config.num_heads,
            "dropout":config.dropout,
            "optimizer_state_dict":optimizer.state_dict(),
            "step":step,
            "best_val_loss":best_val_loss,}
    
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(checkpoint_path)


def append_log(path,step,train_loss,val_loss,best_val_loss):
    log_path = Path(path)
    log_path.parent.mkdir(exist_ok=True)

    file_exists = log_path.exists()

    with log_path.open("a",newline="",encoding="utf-8") as f:
        writer =csv.writer(f)

        if not file_exists:
            writer.writerow(["step","train_loss","val_loss","best_val_loss"])
        writer.writerow([step,train_loss,val_loss,best_val_loss])

def main():
    args = parse_args()
    config = apply_args(TrainConfig(), args)
    save_config_snapshot(config)
    
    text = load_text(config.input_path)
    tokenizer = CharTokenizer(text=text)

    ids = tokenizer.encode(text=text)
    data = torch.tensor(ids,dtype = torch.long)

    train_data,val_data = train_val_split(data=data)

    checkpoint = None
    start_step = 0
    best_val_loss = float("inf")
    
    if config.resume and Path(config.resume_path).exists():
        checkpoint = torch.load(
            config.resume_path,
            map_location=device,
            weights_only=False,
    )

    config.block_size = checkpoint["block_size"]
    config.n_embed = checkpoint["n_embed"]
    config.n_layer = checkpoint["n_layer"]
    config.num_heads = checkpoint["num_heads"]
    config.dropout = checkpoint["dropout"]

    start_step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"resume from {config.resume_path} at step {start_step}")
    model = TinyGPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embed=config.n_embed,
        block_size=config.block_size,
        n_layer=config.n_layer,
        num_heads=config.num_heads,
        dropout=config.dropout,
    )
    model = model.to(device)
    print("Model device:", next(model.parameters()).device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    
    end_step = start_step + config.train_steps
    print(f"training from step {start_step} to {end_step}")
    for step in range(start_step,end_step):
        if step % config.eval_interval ==0:
            losses = estimate_loss(model=model,train_data=train_data,val_data=val_data,config=config)
            print(
                f"step:{step}"
                f" train loss {losses['train']:.4f}"
                f" val loss {losses['val']:.4f}"
            )
            val_loss = losses["val"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    config.best_checkpoint_path,
                    model,
                    optimizer,
                    step + 1,
                    tokenizer,
                    config,
                    best_val_loss,
                )
                print(f"saved best checkpoint to {config.best_checkpoint_path}")

            append_log(config.log_path,step,losses["train"],losses["val"],best_val_loss,)

        x,y = get_batch(train_data,config.batch_size,config.block_size)
        x = x.to(device)
        y = y.to(device)

        logits,loss = model(x,y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    save_checkpoint(
        config.checkpoint_path,
        model,
        optimizer,
        step + 1,
        tokenizer,
        config,
        best_val_loss,
    )

    print(f"saved checkpoint to {config.checkpoint_path}")
    
    

if __name__=="__main__":
    main()