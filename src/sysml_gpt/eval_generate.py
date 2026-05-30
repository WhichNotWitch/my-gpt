"""使用固定prompt检测训练效果"""

import argparse
from pathlib import Path


import torch

from sysml_gpt.model import TinyGPTLanguageModel

def load_prompts(path:str) ->list[str]:
    text = Path(path).read_text(encoding="utf-8")

    prompts=[p.strip("\n") for p in text.split("---")]
    return [p for p in prompts if p.strip()]


def load_model(checkpoint_path:str,device:str):
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False,)

    model = TinyGPTLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        n_embed=checkpoint["n_embed"],
        block_size=checkpoint["block_size"],
        n_layer=checkpoint["n_layer"],
        num_heads=checkpoint["num_heads"],
        dropout=checkpoint["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model,checkpoint

def generate_one(
    model,
    checkpoint,
    prompt:str,
    max_new_tokens:int,
    temperature:float,
    top_k:int|None,
    device:str,
):
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    ids = [stoi[ch] for ch in prompt]
    idx = torch.tensor([ids],dtype=torch.long,device=device)

    generated = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return "".join(itos[i] for i in generated[0].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",default="runs/data_pipeline_test/best.pt")
    parser.add_argument("--prompts",default="eval_prompts.txt")
    parser.add_argument("--output",default="runs/eval_sample.txt")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, checkpoint = load_model(args.checkpoint, device)
    prompts = load_prompts(args.prompts)

    outputs = []

    for i, prompt in enumerate(prompts, start=1):
        text = generate_one(
            model=model,
            checkpoint=checkpoint,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

        outputs.append(
            f"=== Prompt {i} ===\n"
            f"{prompt}\n\n"
            f"=== Output {i} ===\n"
            f"{text}\n"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(outputs), encoding="utf-8")

    print(f"wrote {len(outputs)} samples to {output_path}")


if __name__ == "__main__":
    main()