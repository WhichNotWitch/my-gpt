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
    
if __name__ == "__main__":

    
    text = "package Test {}"
    tokenizer = CharTokenizer(text)
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(ids)
    print(decoded)
    print(tokenizer.vocab_size)