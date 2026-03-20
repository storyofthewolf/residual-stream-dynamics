# Multi-Model Setup for SAELens Exploration

This refactored setup lets you easily swap between models (Pythia, Llama, Gemma, GPT-2) to compare their behavior with your SAE analysis pipeline.

## Quick Start

### 1. List available models
```bash
python predict.py --list-models
```

Output shows which models have pre-trained SAEs available:
```
Available models for SAE exploration:

  pythia-160m          ✓ SAE available           Pythia 160M (deduped, interpretability-focused)
  pythia-1b            ✓ SAE available           Pythia 1B (deduped)
  pythia-3b            ✓ SAE available           Pythia 3B (deduped)
  llama-3.2-1b         ✗ No SAE (train your own) Llama 3.2 1B (limited pre-trained SAEs)
  llama-3.2-3b         ✗ No SAE (train your own) Llama 3.2 3B (limited pre-trained SAEs)
  gemma-2b             ✓ SAE available           Gemma 2B
  gemma-2b-it          ✓ SAE available           Gemma 2B Instruct-tuned
  gpt2-small           ✓ SAE available           GPT-2 Small
  gpt2-medium          ✓ SAE available           GPT-2 Medium
```

### 2. Run with different models

**Compare same prompt across models:**
```bash
# Pythia 1B
python predict.py "2 + 2 =" --model pythia-1b --nt 5

# Llama 3.2 3B (better capability, no SAE available)
python predict.py "2 + 2 =" --model llama-3.2-3b --nt 5

# Gemma 2B (good SAE coverage)
python predict.py "2 + 2 =" --model gemma-2b --nt 5

# Your original GPT-2 small
python predict.py "2 + 2 =" --model gpt2-small --nt 5
```

### 3. Control layer selection

Each model has a default layer for SAE extraction:
```bash
# Use default layer for pythia-1b (layer 6)
python predict.py "Hello" --model pythia-1b

# Override with specific layer
python predict.py "Hello" --model pythia-1b --layer 8

# Gemma 2B instruct default is layer 12
python predict.py "Hello" --model gemma-2b-it --layer 12
```

### 4. Verbose mode for introspection

See token-by-token generation with predictions at each step:
```bash
python predict.py "2 + 2 =" --model pythia-1b --verbose --nt 3
```

## Recommendation: Start with These Models

**For learning SAELens methodology:**

1. **Pythia 1B or 3B** (best for interpretability work)
   - Built by interpretability researchers
   - Pre-trained SAEs available
   - Predictable behavior
   - Good for understanding SAE mechanics

2. **Gemma 2B** (alternative with good SAE coverage)
   - Better performance than Pythia at same scale
   - Excellent pre-trained SAE availability
   - Works well with your pipeline

**For checking what's possible:**

3. **Llama 3.2 3B** (better capability, no SAEs yet)
   - Can load without SAE to see capability baseline
   - Useful for understanding why GPT-2 struggles

```bash
# Compare: Pythia (with SAE) vs Llama (no SAE) on same task
python predict.py "The capital of France is" --model pythia-1b
python predict.py "The capital of France is" --model llama-3.2-3b
```

## How to Use with Your Three-Lens Pipeline

### Update your step1, step2, step3 scripts

At the top of each script, replace your hardcoded model loading with:

```python
import argparse
from setup import load_model_and_sae

# Add to your argument parser
parser.add_argument('--model', type=str, default='pythia-1b')
parser.add_argument('--layer', type=int, default=None)
args = parser.parse_args()

# Load model
model, sae, cfg = load_model_and_sae(
    args.model,
    layer=args.layer,
    load_sae=True
)

# Now use model and sae as usual
hook_name = cfg['hook_name']
```

Then run:
```bash
# Run step1 (logit lens) on Pythia 1B
python step1_logit_lens.py --model pythia-1b

# Run step1 on Gemma 2B for comparison
python step1_logit_lens.py --model gemma-2b
```

## Understanding Pre-trained SAE Availability

**✓ SAEs available** (can use immediately):
- Pythia (160M through 12B)
- Gemma 2B, 2B-IT
- GPT-2 small, medium

**✗ Limited/No SAEs** (need to train your own):
- Llama 3.2 (models are new; SAEs exist in research papers but not in standard registry)
- Mistral 7B

To train an SAE for Llama 3.2:
1. Work through the SAELens tutorial: `training_a_sparse_autoencoder.ipynb`
2. Change model name from "tiny-stories-1L-21M" to "meta-llama/Llama-3.2-3B"
3. Run training pipeline (takes a while on M3 Max, but doable)

## Device Selection

The setup automatically:
- Prefers CPU over MPS (MPS has reliability issues with SAE ops)
- Falls back to CUDA if available
- Can override with `--device cuda` or `--device cpu`

```bash
# Force CPU (most stable on M3 Max)
python predict.py "2 + 2 =" --model pythia-1b --device cpu

# Force MPS if you want to test
python predict.py "2 + 2 =" --model pythia-1b --device mps
```

## Expected Behavior by Model

| Model | Arithmetic | QA | Reasoning | Notes |
|-------|-----------|----|-----------|----|
| Pythia 1B/3B | Sometimes correct | Poor | Poor | Interpretability focus |
| Llama 3.2 3B | Better | OK | OK | Better capability baseline |
| Gemma 2B | Better | Better | Better | Good SAE coverage |
| GPT-2 Small | Wrong | Gibberish | Gibberish | Baseline (struggles) |

Use this to calibrate your interpretability analysis: features you find should correlate with actual model capability.
