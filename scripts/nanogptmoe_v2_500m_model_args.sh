# Megatron/Slim model arguments for nanoGPTMoE-v2 500M.
# Source this file in a launch script to populate MODEL_ARGS.

MODEL_ARGS=(
   --num-layers 16
   --hidden-size 512
   --ffn-hidden-size 2048
   --num-attention-heads 8
   --group-query-attention
   --num-query-groups 4
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 10000
   --vocab-size 151936
   --kv-channels 64
   --untie-embeddings-and-output-weights
   --swiglu
   --moe-layer-freq '[0]+[1]*15'
   --num-experts 12
   --moe-router-topk 3
   --moe-ffn-hidden-size 1024
   --moe-shared-expert-intermediate-size 2048
   --position-embedding-type rope
   --rotary-percent 1.0
)
