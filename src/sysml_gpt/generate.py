"""生成脚本"""

import argparse
import torch
from sysml_gpt.model import TinyGPTLanguageModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default = "checkpoints/tiny_gpt.pt")
    parser.add_argument("--max-new-tokens",type=int,default=300)
    parser.add_argument("--start",default="\n")
    args=parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(args.checkpoint,map_location=device)

    vocab_size = checkpoint["vocab_size"]
    stoi=checkpoint["stoi"]
    itos=checkpoint["itos"]
    block_size = checkpoint["block_size"]
    n_embed = checkpoint["n_embed"]

    model =TinyGPTLanguageModel(vocab_size=vocab_size,
                                n_embed=n_embed,
                                block_size=block_size,
                                )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    start_ids = [stoi[ch] for ch in args.start]
    idx = torch.tensor([start_ids],dtype=torch.long,device=device)

    generated = model.generate(idx,max_new_tokens = args.max_new_tokens)
    text = "".join(itos[i] for i in generated[0].tolist())

    print(text)

if __name__=="__main__":
    main()