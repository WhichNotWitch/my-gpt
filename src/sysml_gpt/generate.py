"""生成脚本"""

import argparse
import torch
from sysml_gpt.model import TinyGPTLanguageModel
from sysml_gpt.tokenizer import tokenizer_from_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default = "checkpoints/tiny_gpt.pt")
    parser.add_argument("--max-new-tokens",type=int,default=300)
    parser.add_argument("--start",default="\n")
    parser.add_argument("--temperature",type = float,default=0.8)
    parser.add_argument("--seed",type=int,default=None)
    parser.add_argument("--top-k",type=int,default=None)
    args=parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
        )

    vocab_size = checkpoint["vocab_size"]
    tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
    block_size = checkpoint["block_size"]
    n_embed = checkpoint["n_embed"]
    n_layer = checkpoint["n_layer"]
    num_heads = checkpoint["num_heads"]
    dropout = checkpoint["dropout"]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model =TinyGPTLanguageModel(vocab_size=vocab_size,
                                n_embed=n_embed,
                                block_size=block_size,
                                n_layer=n_layer,
                                num_heads=num_heads,
                                dropout=dropout,
                                )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    start_ids = tokenizer.encode(args.start)
    idx = torch.tensor([start_ids],dtype=torch.long,device=device)

    generated = model.generate(idx,max_new_tokens = args.max_new_tokens,temperature=args.temperature,top_k=args.top_k)
    text = tokenizer.decode(generated[0].tolist())

    print(text)

if __name__=="__main__":
    main()