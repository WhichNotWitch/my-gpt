"""获取数据并分词"""
from pathlib import Path
import torch

from sysml_gpt.tokenizer import CharTokenizer

def load_text(path:str)->str:
    return Path(path).read_text(encoding="utf-8")

def train_val_split(data:torch.Tensor,train_ratio:float = 0.8):
    n = int(len(data)*train_ratio)
    train_data = data[:n]
    val_data = data[n:]
    return train_data,val_data

def get_batch(data:torch.Tensor,batch_size:int,block_size:int):
    ix = torch.randint(len(data)-block_size,(batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x,y


if __name__== "__main__":
    text = load_text("input.txt")
    tokenizer= CharTokenizer(text)

    ids = tokenizer.encode(text)
    data = torch.tensor(ids,dtype = torch.long)

    train_data ,val_data = train_val_split(data)
    x,y = get_batch(train_data,batch_size=4,block_size=8)

    print("text length:", len(text))
    print("vocab size:", tokenizer.vocab_size)
    print("data shape:", data.shape)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x[0]:", x[0])
    print("y[0]:", y[0])
    print("x[0] decoded:")
    print(tokenizer.decode(x[0].tolist()))
    print("y[0] decoded:")
    print(tokenizer.decode(y[0].tolist()))