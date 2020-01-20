import os
import torch

TEST_DIR = '/data/train'


def inference():
    print()
    print("[TEST FILES]")
    for file in sorted(os.listdir(TEST_DIR)):
        print("  ", file)
    print()
    print()


if __name__ == "__main__":
    inference()
