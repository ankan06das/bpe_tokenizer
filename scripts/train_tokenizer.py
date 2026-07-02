"""
Train the BPE tokenizer from a .txt corpus and save as .pkl.

Usage:
    python scripts/train_tokenizer.py --corpus data/corpus.txt --vocab_size 37000 --save models/tokenizer.pkl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.pretokenizer import PreBPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer from scratch")
    parser.add_argument("--corpus", required=True, help="Path to training corpus (.txt)")
    parser.add_argument("--vocab_size", type=int, default=37000, help="Vocabulary size (incl. 256 byte tokens)")
    parser.add_argument("--save", default="models/tokenizer.pkl", help="Output path for .pkl model")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: corpus file not found: {corpus_path}")
        sys.exit(1)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    print(f"Corpus size: {len(corpus):,} chars")
    print(f"Target vocab size: {args.vocab_size:,} ({args.vocab_size - 256:,} merges)")

    tokenizer = PreBPETokenizer(num_workers=args.workers)
    tokenizer.train(corpus, args.vocab_size)
    tokenizer.save(args.save)

    print(f"Done! Model saved to {args.save}")
    print(f"  Merge rules:  {len(tokenizer.merge_rules):,}")
    print(f"  Vocab size:   {len(tokenizer.vocab):,}")



if __name__ == "__main__":
    main()
