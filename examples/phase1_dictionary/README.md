# Phase-1 Dictionary Example

This directory contains a small excerpt of real dictionary entries from the
project's local phase-1 dictionary corpus for testing the pretraining data
pipeline. It is not the full training corpus. The raw files use
`<|endoftext|>` as the document separator expected by
`scripts/tokenize_corpus.py`.

The excerpt is provided as a dataset example. Check the upstream terms and
licenses before redistributing a larger corpus or using additional source files.

Run from the repository root:

```bash
python -u -m scripts.tokenize_corpus \
  --config configs/nanogptmoe_v2_500m_liger.yaml \
  --input-dir examples/phase1_dictionary/raw \
  --output-dir examples/phase1_dictionary/tokenized_qwen3_ctx2048 \
  --context-length 2048
```

Then use `configs/examples/train_pretraining.yaml`. Its `data_dir` already
points to the generated output directory. The generated binary files are local
artifacts and are intentionally not committed.
