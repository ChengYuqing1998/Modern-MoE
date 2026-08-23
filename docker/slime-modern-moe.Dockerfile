FROM slimerl/slime:latest

# The official image omits the tokenizer backends needed by the project's
# Qwen2 tokenizer and Megatron HuggingFaceTokenizer wrapper.
RUN python -m pip install --no-cache-dir tiktoken sentencepiece
