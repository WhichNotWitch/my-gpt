"""简易字符级tokenizer"""
class CharTokenizer:
    def __init__(self,text:str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch : i for i,ch in enumerate(chars)}
        self.itos = {i : ch for i,ch in enumerate(chars)}

    def encode(self,text:str)-> list[int]:
        return [self.stoi[ch] for ch in text] 

    def decode(self,ids:list[int])->str:
        return "".join(self.itos[id] for id in ids)       
    
    def state_dict(self) -> dict:
        return {
            "type": "char",
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": self.itos,
        }

    
"""字符级BPEtokenizer"""
def get_pair_counts(tokens:list[str])->dict[tuple[str,str],int]:
    counts={}

    for a,b in zip(tokens,tokens[1:]):
        pair = (a,b)
        counts[pair] = counts.get(pair,0)+1

    return counts

def merge_pair(tokens:list[str],pair:tuple[str,str],new_tokens:str)->list[str]:
    merged = []
    i =0

    while i<len(tokens):
        if i <len(tokens) -1 and (tokens[i],tokens[i+1]) == pair:
            merged.append(new_tokens)
            i+=2
        else :
            merged.append(tokens[i])
            i+=1
    return merged

class CharBPETokenizer:
    def __init__(self,vocab_size:int=500):
        self.target_vocab_size =vocab_size
        self.merges : list[tuple[str,str]]=[]
        self.token_to_id:dict[str,int] = {}
        self.id_to_token:dict[int,str] ={}
        self.vocab_size =0

    def train(self,text:str):
        tokens = list(text)

        vocab = set(tokens)

        while len(vocab) <self.target_vocab_size:
            pair_counts = get_pair_counts(tokens=tokens)

            if not pair_counts:
                break

            best_pair = max(pair_counts,key=pair_counts.get)

            new_token = best_pair[0]+best_pair[1]

            if new_token in vocab:
                break

            tokens = merge_pair(tokens=tokens,pair=best_pair,new_tokens=new_token)
            self.merges.append(best_pair)
            vocab.add(new_token)

        sorted_vocab = sorted(vocab)

        self.token_to_id = {token:i for i,token in enumerate(sorted_vocab) }
        self.id_to_token = {i: token for i, token in self.token_to_id.items()}

        self.vocab_size = len(self.token_to_id)
    
    def encode(self, text: str) -> list[int]:
        tokens = list(text)

        for pair in self.merges:
            new_token = pair[0] + pair[1]
            tokens = merge_pair(tokens, pair, new_token)

        return [self.token_to_id[token] for token in tokens]


    def decode(self, ids: list[int]) -> str:
        tokens = [self.id_to_token[i] for i in ids]
        return "".join(tokens)
    

"""字符级BPEtokenizer"""
def get_byte_pair_counts(tokens: list[int]) -> dict[tuple[int, int], int]:
    counts = {}

    for a, b in zip(tokens, tokens[1:]):
        pair = (a, b)
        counts[pair] = counts.get(pair, 0) + 1

    return counts

def merge_byte_pair(
    tokens: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    merged = []
    i = 0

    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1

    return merged

class ByteBPETokenizer:
    def __init__(self, vocab_size: int = 500):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")

        self.target_vocab_size = vocab_size
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size = len(self.vocab)

    def train(self, text: str):
        tokens = list(text.encode("utf-8"))

        next_id = 256

        while next_id < self.target_vocab_size:
            pair_counts = get_byte_pair_counts(tokens)

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)

            if best_pair in self.merges:
                break

            self.merges[best_pair] = next_id
            self.vocab[next_id] = (
                self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            )

            tokens = merge_byte_pair(tokens, best_pair, next_id)
            next_id += 1

        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))

        for pair, new_id in self.merges.items():
            tokens = merge_byte_pair(tokens, pair, new_id)

        return tokens
    
    def decode(self, ids: list[int]) -> str:
        byte_chunks = [self.vocab[i] for i in ids]
        data = b"".join(byte_chunks)
        return data.decode("utf-8", errors="replace")
    
    def state_dict(self)->dict:
        return {
            "type":"byte_bpe",
            "target_vocab_size":self.target_vocab_size,
            "merges":[
                [a,b,new_id]
                for (a,b),new_id in self.merges.items()
            ]
        }
    
    @classmethod
    def from_state_dict(cls,state:dict):
        tokenizer = cls(vocab_size=state["target_vocab_size"])
        tokenizer.merges = {}
        tokenizer.vocab = {i:bytes([i]) for i in range(256)}

        for a,b,new_id in state["merges"]:
            pair = (a,b)
            tokenizer.merges[pair] = new_id
            tokenizer.vocab[new_id] = tokenizer.vocab[a] +tokenizer.vocab[b]

        tokenizer.vocab_size = len(tokenizer.vocab)
        return tokenizer


def build_tokenizer(kind:str,text:str,vocab_size:int|None=None):
    if kind=="char":
        return CharTokenizer(text)
    
    if kind=="char-bpe":
        return None
    
    if kind=="byte-bpe":
        if vocab_size is None:
            raise ValueError("vocab_size if required for byte-bpe tokenizer")
        
        tokenizer = ByteBPETokenizer(vocab_size=vocab_size)
        tokenizer.train(text=text)
        return tokenizer
    
    raise ValueError(f"unknown tokenizer kind: {kind}")
    

def tokenizer_from_state(state:dict):
    tokenizer_type = state["type"]

    if tokenizer_type =="char":
        tokenizer = object.__new__(CharTokenizer)
        tokenizer.stoi = state["stoi"]
        tokenizer.itos = {int(k):v for k,v in state["itos"].items()}
        tokenizer.vocab_size = state["vocab_size"]
        return tokenizer
    
    if tokenizer_type=="byte_bpe":
        return ByteBPETokenizer.from_state_dict(state)
    
    raise ValueError(f"unknown tokenizer type: {tokenizer_type}")
if __name__ == "__main__":
    text = "attribute attribute action action package package 属性 测试"

    tokenizer = ByteBPETokenizer(vocab_size=300)
    tokenizer.train(text)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print("vocab size:", tokenizer.vocab_size)
    print("num utf-8 bytes:", len(text.encode("utf-8")))
    print("num bpe tokens:", len(ids))
    print("roundtrip:", decoded == text)

    print("first merges:")
    for i, (pair, new_id) in enumerate(tokenizer.merges.items()):
        if i >= 20:
            break
        print(pair, "->", new_id, tokenizer.vocab[new_id])

    state = tokenizer.state_dict()
    restored = ByteBPETokenizer.from_state_dict(state)

    restored_ids = restored.encode(text)
    restored_decoded = restored.decode(restored_ids)

    print("restored roundtrip:", restored_decoded == text)
    print("same ids:", restored_ids == ids)