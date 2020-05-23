import sys

from src.eval import eval_dmm

if __name__ == "__main__":
    output_dir = sys.argv[1]
    print(f"Evaluating {output_dir}")
    eval_dmm(output_dir)
