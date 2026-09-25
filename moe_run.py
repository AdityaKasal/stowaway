"""Older name for moe.py, kept so earlier commands still work: python moe_run.py <model.gguf> [options]."""

import sys

from moe import main

if __name__ == "__main__":
    main(["run"] + sys.argv[1:])
