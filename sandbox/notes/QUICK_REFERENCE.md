# Quick Reference: Multi-Model Setup

## One-Liners to Get Started

```bash
# See all available models
python predict.py --list-models

# Test arithmetic on Pythia 1B
python predict.py "2 + 2 =" --model pythia-1b

# Test on Llama 3.2 3B (no SAE available yet)
python predict.py "2 + 2 =" --model llama-3.2-3b

# Test on Gemma 2B (good SAE coverage)
python predict.py "2 + 2 =" --model gemma-2b

# Verbose mode (see generation step-by-step)
python predict.py "2 + 2 =" --model pythia-1b --verbose

# Override layer
python predict.py "Hello" --model pythia-1b --layer 8
```

## Model Selection for Your Goals

**For learning SAELens with solid SAE coverage:**
```bash
python predict.py "2 + 2 =" --model pythia-1b          # Interpretability-focused
python predict.py "2 + 2 =" --model gemma-2b           # Better capability
```

**For comparing capability levels:**
```bash
# This will show why GPT-2 fails where Llama succeeds
python predict.py "The capital of France is" --model gpt2-small
python predict.py "The capital of France is" --model llama-3.2-3b
```

**For three-lens pipeline on different models:**
```bash
# Run your step1, step2, step3 scripts on different models
python step1_logit_lens.py --model pythia-1b
python step1_logit_lens.py --model gemma-2b
python step1_logit_lens.py --model llama-3.2-3b --no-sae
```

## Key Files

- **setup.py** — Model registry & loading. Contains `load_model_and_sae()` function
- **predict.py** — Token prediction with full introspection
- **MODEL_SELECTION_GUIDE.md** — Full documentation

## Model Registry Structure

```python
MODEL_CONFIGS = {
    "pythia-1b": {
        "hf_name": "EleutherAI/pythia-1b-deduped",
        "sae_release": "pythia-1b-res-jb",
        "default_layer": 6,
        "hook_pattern": "blocks.{layer}.hook_resid_post",
    },
    # ... more models
}
```

To add a new model:
1. Add entry to `MODEL_CONFIGS` in setup.py
2. Run `python predict.py --list-models` to verify

## Adding New Models

Example: if you want to add Mistral 7B:

```python
"mistral-7b": {
    "hf_name": "mistralai/Mistral-7B-v0.1",
    "sae_release": None,  # No pre-trained SAE yet
    "default_layer": 10,
    "hook_pattern": "blocks.{layer}.hook_resid_post",
    "description": "Mistral 7B (limited SAE coverage)"
},
```

Then use it:
```bash
python predict.py "Hello" --model mistral-7b --no-sae
```

## Device Notes

- Auto-selects CPU over MPS for stability
- Can override with `--device cuda` or `--device cpu`
- M3 Max: CPU is typically more reliable for SAE ops

## SAE Availability Status

| Model | SAE Available | Quality |
|-------|---------------|---------|
| pythia-160m | ✓ | Good |
| pythia-1b | ✓ | Good |
| pythia-3b | ✓ | Good |
| gemma-2b | ✓ | Excellent |
| gemma-2b-it | ✓ | Excellent |
| gpt2-small | ✓ | Good |
| gpt2-medium | ✓ | Good |
| llama-3.2-1b | ✗ | None (research only) |
| llama-3.2-3b | ✗ | None (research only) |
| mistral-7b | ✗ | Limited (research papers) |

## Common Workflows

### Compare Arithmetic Behavior
```bash
for model in pythia-1b pythia-3b gemma-2b llama-3.2-3b; do
    echo "=== $model ==="
    python predict.py "2 + 2 =" --model $model --nt 3
done
```

### Find Which Layer Works Best
```bash
for layer in 4 6 8 10; do
    echo "=== Layer $layer ==="
    python predict.py "2 + 2 =" --model pythia-1b --layer $layer --nt 1
done
```

### Integrate with Your Pipeline
```python
# In step1_logit_lens.py, step2_max_activating.py, step3_causal_patch.py:
# Add at top:
import argparse
from setup import load_model_and_sae

parser.add_argument('--model', type=str, default='pythia-1b')
args = parser.parse_args()

model, sae, cfg = load_model_and_sae(args.model)
hook_name = cfg['hook_name']
```

Then:
```bash
python step1_logit_lens.py --model pythia-1b
python step1_logit_lens.py --model gemma-2b
# ... run both, compare results
```
