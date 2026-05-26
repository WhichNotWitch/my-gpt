"""gpt模型文件"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    """最基础的语言模型"""
    def __init__(self,vocab_size:int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx:torch.Tensor,targets:torch.Tensor|None = None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            batch_size,block_size,vocab_size = logits.shape

            logits = logits.view(batch_size*block_size,vocab_size)  
            targets = targets.view(batch_size*block_size)

            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self,idx:torch.Tensor,max_new_tokens:int):
        for _ in range(max_new_tokens):
            logits,_ = self(idx)

            logits = logits[:,-1,:]

            probs = F.softmax(logits,dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
class Head(nn.Module):
    def __init__(self,n_embed:int,head_size:int,block_size:int):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias=False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        
        self.register_buffer("tril",
                             torch.tril(torch.ones(block_size,block_size)),
                             )


    def forward(self,x:torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2,-1)
        weights = weights * k.shape[-1] **-0.5


        weights = weights.masked_fill(
            self.tril[:T,:T] ==0,
            float("-inf")
        )

        weights = F.softmax(weights,dim=-1)

        v = self.value(x)
        out = weights @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_embed:int,
            num_heads:int,
            head_size:int,
            block_size:int,
            ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(
                n_embed,
                head_size,
                block_size,
            ) for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_heads *head_size,n_embed)

    def forward(self,x:torch.Tensor):
        out = torch.cat([head(x) for head in self.heads],dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self,n_embed:int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed),
        )

    def forward(self,x:torch.Tensor):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embed:int,num_heads:int,block_size:int):
        super().__init__()

        head_size = n_embed // num_heads

        self.sa = MultiHeadAttention(
            n_embed,
            num_heads,
            head_size,
            block_size,
        )
        self.ffwd = FeedForward(n_embed)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x:torch.Tensor):
        #很多 GPT 风格模型更常用 Pre-LN，因为训练深层 Transformer 时更稳定。
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPTLanguageModel(nn.Module):
    """带位置编码的语言模型"""
    def __init__(self,vocab_size:int,n_embed:int,block_size:int):
        super().__init__()

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(block_size,n_embed)
        #self.sa_head = Head(n_embed,n_embed,block_size)
        num_heads = 4

        self.blocks=nn.Sequential(
            Block(
                n_embed,
                num_heads,
                block_size,
            )
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)
        

    def forward(self,idx:torch.Tensor,targets:torch.Tensor | None=None):
        batch_size,block_size = idx.shape

        token_emb = self.token_embedding_table(idx)

        positions = torch.arange(block_size,device=idx.device)
        pos_emb=self.position_embedding_table(positions)

        x = token_emb+pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits=self.lm_head(x)

        if(targets==None):
            loss=None
        else:
            batch_size,block_size,vocab_size = logits.shape
            logits = logits.view(batch_size*block_size,vocab_size)
            targets = targets.view(batch_size*block_size,)
            loss = F.cross_entropy(logits,targets)

        return logits,loss
    
    def generate(self,idx:torch.Tensor,max_new_tokens:int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]
            logits,_ = self(idx_cond)

            logits = logits[:,-1,:]

            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)

        return idx


"""BigramLanguageModel的测试
if __name__ == "__main__":
    model = BigramLanguageModel(vocab_size=79)

    x = torch.randint(0, 79, (4, 8))
    y = torch.randint(0, 79, (4, 8))

    logits, loss = model(x, y)

    print("logits shape:", logits.shape)
    print("loss:", loss.item())

    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=20)

    print("generated shape:", generated.shape)
    print(generated)
"""

if __name__ == "__main__":
    vocab_size = 79
    block_size = 8
    n_embed = 32

    model = TinyGPTLanguageModel(
        vocab_size=vocab_size,
        n_embed=n_embed,
        block_size=block_size,
    )

    x = torch.randint(0, vocab_size, (4, block_size))
    y = torch.randint(0, vocab_size, (4, block_size))

    logits, loss = model(x, y)

    print("logits shape:", logits.shape)
    print("loss:", loss.item())

    start = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start, max_new_tokens=20)

    print("generated shape:", generated.shape)
    print(generated)

