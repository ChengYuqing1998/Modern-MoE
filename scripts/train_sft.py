"""SFT entrypoint.

The optimized training implementation remains shared with ``scripts.train``;
the SFT YAML selects assistant-only mmap data and disables the incompatible
all-targets-valid CUDA Graph path.
"""

from scripts.train import main


if __name__ == "__main__":
    main()
