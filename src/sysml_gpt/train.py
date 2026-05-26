"""训练"""


from pathlib import Path

import torch

from sysml_gpt.data import get_batch,load_text,train_val_split
from sysml_gpt.model import TinyGPTLanguageModel
from sysml_gpt.tokenizer import CharTokenizer

batch_size = 32
block_size = 64
max_iters = 2000
eval_interval = 100
learning_rate = 3e-4
n_embed = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def estimate_loss(model,train_data,val_data):
    out ={}
    model.eval()

    for split,data in [("train",train_data),("val",val_data)]:
        losses = torch.zeros(20)

        for k in range(20):
            x,y = get_batch(data,batch_size=batch_size,block_size=block_size)
            x = x.to(device)
            y = y.to(device)

            _,loss = model(x,y)
            # loss是一个tensor，里面存了计算图等其他信息，用.item()仅仅取出数值
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out

def main():
    text = load_text("input.txt")
    tokenizer = CharTokenizer(text=text)

    ids = tokenizer.encode(text=text)
    data = torch.tensor(ids,dtype = torch.long)

    train_data,val_data = train_val_split(data=data)

    model = TinyGPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embed=n_embed,
        block_size=block_size,
    )
    model = model.to(device)

    optimzer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval ==0:
            losses = estimate_loss(model=model,train_data=train_data,val_data=val_data)
            print(
                f"step:{step}"
                f" train loss {losses['train']:.4f}"
                f" val loss {losses['val']:.4f}"
            )

        x,y = get_batch(train_data,batch_size,block_size)
        x = x.to(device)
        y = y.to(device)

        logits,loss = model(x,y)

        optimzer.zero_grad(set_to_none=True)
        loss.backward()
        optimzer.step()

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir/"tiny_gpt.pt"

    torch.save(
        {
            "model_state_dict":model.state_dict(),
            "vocab_size":tokenizer.vocab_size,
            "stoi":tokenizer.stoi,
            "itos":tokenizer.itos,
            "block_size":block_size,
            "n_embed":n_embed
        },
        checkpoint_path,
    )

    print(f"saved checkpoint to {checkpoint_path}")

if __name__=="__main__":
    main()