# Llama-2 Quick Start Guide

## Installation

```bash
# Install FlexLLMGen with dependencies
pip install sentencepiece transformers torch

# Clone and setup
cd FlexLLMGen2
pip install -e .
```

## Basic Usage

### Test with Dummy Weights

```bash
python -m flexllmgen.flex_llama \
    --model meta-llama/Llama-2-7b-chat \
    --path _DUMMY_
```

### Use Real Weights

**Step 1**: Request access to Llama-2
- Visit: https://huggingface.co/meta-llama/Llama-2-7b-chat
- Accept Meta's license agreement

**Step 2**: Login to HuggingFace
```bash
huggingface-cli login
```

**Step 3**: Run with automatic weight download
```bash
python -m flexllmgen.flex_llama \
    --model meta-llama/Llama-2-7b-chat \
    --path ./llama-2-7b-np
```

Weights will be automatically downloaded and converted on first run.

## Command-Line Options

```bash
python -m flexllmgen.flex_llama \
    --model meta-llama/Llama-2-7b-chat \  # Model name
    --path _DUMMY_ \                       # Weight path (_DUMMY_ for testing)
    --gpu-batch-size 32 \                  # Batch size
    --percent 100 0 100 0 100 0 \         # Memory policy (see below)
    --compress-weight                      # Enable weight compression
```

### Memory Policy (`--percent`)

Format: `<w_gpu%> <w_cpu%> <w_disk%> <c_gpu%> <c_cpu%> <c_disk%>`

**Examples**:
```bash
# All on GPU (fastest, needs ~14GB GPU memory)
--percent 100 0 0 0 100 0

# 50% weights on CPU (reduces GPU memory to ~7GB)
--percent 50 50 0 0 100 0

# All weights on disk, cache on GPU (minimal GPU memory)
--percent 0 0 100 0 100 0

# Balanced CPU offloading
--percent 50 50 0 0 50 50
```

## Expected Output

```
<run_flexllmgen>: args.model: meta-llama/Llama-2-7b-chat
model size: 12.308 GB, cache size: 1.062 GB, hidden size (prefill): 0.017 GB
init weight...
warmup - generate
benchmark - generate
TorchDevice: cuda:0
  cur_mem: 12.5604 GB,  peak_mem: 13.8253 GB
model size: 12.308 GB   cache size: 1.062 GB
peak gpu mem: 13.825 GB
prefill latency: 32.015 s       prefill throughput: 63.970 token/s
decode latency: 25.084 s        decode throughput: 4.943 token/s
total latency: 57.099 s         total throughput: 2.242 token/s
```

## Model Variants

```python
# Supported models
"meta-llama/Llama-2-7b"          # Base 7B
"meta-llama/Llama-2-7b-chat"     # Chat-tuned 7B
"meta-llama/Llama-2-13b"         # Base 13B
"meta-llama/Llama-2-13b-chat"    # Chat-tuned 13B
"meta-llama/Llama-2-70b"         # Base 70B (needs offloading)
"meta-llama/Llama-2-70b-chat"    # Chat-tuned 70B (needs offloading)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **401 Unauthorized** | Request access on HuggingFace Hub |
| **Out of Memory** | Use CPU offloading: `--percent 50 50 0 0 100 0` |
| **Slow generation** | Increase GPU allocation: `--percent 100 0 0 0 100 0` |
| **Tokenizer errors** | Already fixed (uses slow tokenizer) |

## Performance Tips

**For 7B model**:
- Use `--percent 100 0 0 0 100 0` if you have 16GB+ GPU
- Use `--percent 50 50 0 0 100 0` for 8-12GB GPUs

**For 13B model**:
- Requires CPU offloading: `--percent 30 70 0 0 100 0`
- Or use compression: `--compress-weight`

**For 70B model**:
- Heavy offloading required: `--percent 10 40 50 0 50 50`
- Consider multi-GPU setup (future feature)

## Files Created

- `flexllmgen/llama_config.py` - Model configurations
- `flexllmgen/flex_llama.py` - Main implementation
- `docs/llama2_implementation.md` - Detailed documentation

## Next Steps

1. **Validate with real weights**: Test generation quality
2. **Benchmark performance**: Compare with HuggingFace Transformers
3. **Explore offloading**: Find optimal memory policies for your hardware

For detailed information, see `docs/llama2_implementation.md`
