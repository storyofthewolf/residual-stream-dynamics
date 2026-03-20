"""
Step 1: Logit Lens
For each feature, project its SAE decoder direction into vocabulary space
to see which tokens it promotes/suppresses. Fast semantic labels.
"""

# captures import of python files from one directory up
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse
import torch
from setup import load_model_and_sae

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pythia-1b')
parser.add_argument('--layer', type=int, default=None)
args = parser.parse_args()

model, sae, cfg = load_model_and_sae(
    args.model,
    layer=args.layer,
    load_sae=True
)

hook_name = cfg['hook_name']



FEATURE_INDICES = [12645, 2621, 6791, 15822, 3886, 16915, 18197, 11668, 7467, 15844]
TOP_K = 10

print("=" * 60)
print("LOGIT LENS: What does each feature promote/suppress?")
print("=" * 60)

# Move to CPU — MPS can be finicky with large matmuls
# W_dec: [num_features, d_model]  W_U: [d_model, vocab_size]
W_dec = sae.W_dec.to("cpu")
W_U   = model.W_U.to("cpu")

for feat_idx in FEATURE_INDICES:
    direction     = W_dec[feat_idx]          # [d_model]
    logit_scores  = direction @ W_U          # [vocab_size]

    top_vals, top_ids = torch.topk(logit_scores, k=TOP_K)
    bot_vals, bot_ids = torch.topk(logit_scores, k=TOP_K, largest=False)

    top_tokens = [model.to_string([i.item()]) for i in top_ids]
    bot_tokens = [model.to_string([i.item()]) for i in bot_ids]

    print(f"\nFeature {feat_idx}")
    print(f"  PROMOTES   → {list(zip(top_tokens, [round(v,2) for v in top_vals.tolist()]))}")
    print(f"  SUPPRESSES → {list(zip(bot_tokens, [round(v,2) for v in bot_vals.tolist()]))}")

print("""
INTERPRETATION GUIDE
  Digit tokens ('4', ' 4', '8')    → answer/output feature
  Operator tokens ('+', '-', '=')  → structural/positional feature
  Mixed arithmetic tokens           → general 'math context' feature
  Unrelated tokens                  → co-activation artifact, likely not causal
""")
