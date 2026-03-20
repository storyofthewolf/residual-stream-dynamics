"""
Step 2: Max Activating Examples
Build a corpus of arithmetic prompts, run them all through the model,
and rank which examples most strongly activate each feature.
Confirms (or challenges) the Step 1 labels with real evidence.
"""
import torch
from load_model import model, sae, device

FEATURE_INDICES = [12645, 2621, 6791, 15822, 3886, 16915, 18197, 11668, 7467, 15844]
TOP_K = 5

# Diverse corpus: vary operator, operand magnitude, and include controls
corpus = []
for a in range(0, 20):
    for b in range(0, 20):
        corpus.append(f"{a} + {b} =")
for a in range(0, 20):
    for b in range(0, a + 1):
        corpus.append(f"{a} - {b} =")
for a in range(2, 10):
    for b in range(2, 10):
        corpus.append(f"{a} * {b} =")
for a in range(10, 30):
    corpus.append(f"{a} + 5 =")
    corpus.append(f"{a} - 3 =")

# Non-arithmetic controls — do features fire here too?
corpus += [
    "The cat sat on the",
    "Once upon a time there",
    "The capital of France is",
    "She walked into the room and",
    "def add(a, b): return",
]

print(f"Corpus: {len(corpus)} prompts")
print("Running forward passes... (~60s on MPS)")

feature_to_examples = {idx: [] for idx in FEATURE_INDICES}

for prompt in corpus:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        hidden = cache["blocks.6.hook_resid_pre"]  # [1, seq_len, d_model]

    # Last token position — where next-token prediction happens
    last_hidden  = hidden[0, -1, :]       # [d_model]
    features     = sae.encode(last_hidden).cpu()  # [num_features]

    for feat_idx in FEATURE_INDICES:
        act = features[feat_idx].item()
        if act > 0:   # SAE uses ReLU; zero = feature is off
            feature_to_examples[feat_idx].append((act, prompt))

print("\n" + "=" * 60)
print("MAX ACTIVATING EXAMPLES PER FEATURE")
print("=" * 60)

for feat_idx in FEATURE_INDICES:
    examples = sorted(feature_to_examples[feat_idx], reverse=True)[:TOP_K]
    total    = len(feature_to_examples[feat_idx])
    print(f"\nFeature {feat_idx}  ({total} activations in corpus)")
    if not examples:
        print("  [No activations — feature may be inactive for this prompt type]")
    for act, prompt in examples:
        print(f"  {act:6.3f}  |  {repr(prompt)}")

print("""
INTERPRETATION GUIDE
  Step 1 (logit lens) and Step 2 (examples) should AGREE for clean features.
  Feature fires on '5+3' but not '10-2'  → addition-specific
  Feature fires on all arithmetic         → general math context
  Feature fires on controls too           → probably not arithmetic-specific
""")
