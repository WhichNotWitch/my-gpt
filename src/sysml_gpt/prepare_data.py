"""数据预处理"""

import argparse
from pathlib import Path

SEPARATOR = "<|endoftext|>"

def read_text_files(input_dir:Path) -> list[tuple[Path,str]]:
    paths = sorted(
        list(input_dir.glob("*.txt"))
        + list(input_dir.glob("*.sysml"))
        + list(input_dir.glob("*.kerml"))
    )
    files = []

    for path in paths:
        text = path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        files.append((path,text))

    return files

def build_corpus(input_dir:Path,output_path:Path):
    files=read_text_files(input_dir)

    if not files:
        raise ValueError(f"no .txt,.sysml,or .kerml files found in {input_dir}")
    
    chunks = [text for _,text in files]

    corpus = f"\n{SEPARATOR}\n".join(chunks)
    corpus = corpus +"\n"

    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(corpus,encoding="utf-8")

    total_chars = len(corpus)
    print(f"loaded {len(files)} files")
    print(f"wrote {total_chars} characters to {output_path}")

    for path,text in files:
        print(f"- {path}: {len(text)} chars")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",default="data/raw")
    parser.add_argument("--output-path",default="data/processed/train.txt")
    args = parser.parse_args()

    build_corpus(
        input_dir= Path(args.input_dir),
        output_path= Path(args.output_path),
    )

if __name__ == "__main__":
    main()