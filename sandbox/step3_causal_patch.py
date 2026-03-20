"""
Step 3: Causal Patching
Zero out each feature one at a time using a TransformerLens hook,
then measure how much the correct answer token's probability drops.
This proves causation, not just correlation.
"""
import torch
import torch.nn.functional as F
from setup import model, sae, device

FEATURE_INDICES = [12645, 2621, 6791, 15822, 3886, 16915, 18197, 11668, 7467, 15844]

PROMPT        = "2 + 2 ="
ANSWER_TOKEN  = " 4"   # the correct next token (note the leading space)

tokens       = model.to_tokens(PROMPT)
answer_id    = model.to_single_token(ANSWER_TOKEN)

def get_answer_prob(patched_hidden=None):
    """
    Run a forward pass. If patched_hidden is provided, inject it at
    blocks.6.hook_resid_pre instead of the model's own activations.
    Returns the probability assigned to the answer token.
    """
    if patched_hidden is None:
        with torch.no_grad():
            logits = model(tokens)
    else:
        def hook_fn(value, hook):
            return patched_hidden  # swap in our modified activations

        with torch.no_grad():
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[("blocks.6.hook_resid_pre", hook_fn)]
            )

    # logits shape: [1, seq_len, vocab_size]; take last token position
    last_logits = logits[0, -1, :]
    probs       = F.softmax(last_logits, dim=-1)
    return probs[answer_id].item()

# ── Baseline (no patching) ────────────────────────────────────────────────────
with torch.no_grad():
    _, cache   = model.run_with_cache(tokens)
    hidden     = cache["blocks.6.hook_resid_pre"]  # [1, seq_len, d_model]
    last_pos   = hidden[0, -1, :]                  # [d_model]
    features   = sae.encode(last_pos)              # [num_features]

baseline_prob = get_answer_prob()
print(f"Baseline P('{ANSWER_TOKEN}') = {baseline_prob:.4f}")
print("=" * 60)
print("CAUSAL PATCHING: zeroing each feature one at a time")
print("=" * 60)

results = []

for feat_idx in FEATURE_INDICES:
    feat_activation = features[feat_idx].item()

    # Zero this feature and decode back to residual stream space
    patched_features = features.clone()
    patched_features[feat_idx] = 0.0

    # Decode: project modified features back to d_model
    # sae.decode reconstructs the residual stream contribution
    patched_last = sae.decode(patched_features)         # [d_model]

    # Rebuild the full hidden state with the patched last position
    patched_hidden = hidden.clone()
    patched_hidden[0, -1, :] = patched_last

    patched_prob = get_answer_prob(patched_hidden)
    prob_drop    = baseline_prob - patched_prob
    pct_drop     = (prob_drop / baseline_prob) * 100

    results.append((pct_drop, feat_idx, feat_activation, patched_prob))
    print(f"  Feature {feat_idx:5d}  activation={feat_activation:6.3f}  "
          f"P(answer): {baseline_prob:.4f} → {patched_prob:.4f}  "
          f"drop={pct_drop:+.1f}%")

# Sort by causal importance
print("\n" + "=" * 60)
print("RANKED BY CAUSAL IMPORTANCE (largest drop = most important)")
print("=" * 60)
for pct_drop, feat_idx, feat_act, patched_prob in sorted(results, reverse=True):
    bar = "#" * max(0, int(abs(pct_drop) / 2))
    print(f"  Feature {feat_idx:5d}  {pct_drop:+6.1f}%  {bar}")

print("""
INTERPRETATION GUIDE
  Large drop (>10%)  → this feature is causally important for the answer
  Small drop (<2%)   → active but not actually doing the computation here
  Negative drop      → zeroing this feature HELPS (it was suppressing the answer)
  
Next step: for your top causal features, try AMPLIFYING them (scale > 1.0)
and see if the model becomes MORE confident, or shifts to a wrong answer.
""")
