"""字符级tokenizer"""
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