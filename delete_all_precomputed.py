"""
Simple utility scipt to delete precomputed RelGT tokens.
"""

import os
import shutil


def work(path):
    if os.path.isdir(path):
        if os.path.basename(path) == "precomputed":
            print(f"Removing {path}...")
            shutil.rmtree(path)
            return
        else:
            for file in os.listdir(path):
                work(os.path.join(path, file))


def main():
    work(os.path.join("cache", "data"))


if __name__ == "__main__":
    main()
