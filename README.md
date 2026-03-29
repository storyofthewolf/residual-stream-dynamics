# residual stream dynamics

A package for analysising the residual stream and related layer properties of toy LLMs

## File Structure
- extraction.py — defines ActivationRecord and extract_activations(). The critical function accepts a list of hook types and returns a dict of records from one forward pass. The HOOK_TYPES registry at the top is where you add new hook types — no other changes needed elsewhere. The has_resid_mid flag is detected automatically by probing the cache before the main forward pass.
- computation.py — defines EntropyRecord and compute_entropy_surface(). One ActivationRecord → multiple EntropyRecords (one per norm×alpha combination). The filter_records() helper is the primary tool for selecting subsets downstream. Future analysis types (RenyiSpectrumRecord, VonNeumannRecord, DynamicsRecord) follow the same pattern — new dataclass, new compute function, same ActivationRecord input.
- entropy_plots.py — all plot functions now accept EntropyRecord objects directly. Hook type, norm, alpha, d_model, and prompt are all read from the record rather than passed as separate arguments. The new plot_hook_comparison() is the primary tool for your attention vs MLP scientific question.
- workflows/explore_prompt.py — edit DEFAULT_PROMPTS at the top for quick experiments. Runs --hooks resid_post attn_out mlp_out by default, generating the hook comparison plot automatically. Use --save-data to write results for later multi-model comparison.
- workflows/corpus_analysis.py — drops in as a replacement for the old corpus mode in residual_stream_entropy.py. Same --corpus and --model flags as before.


## Quick Start Commands
### Default: gpt2-small, hooks resid_post + attn_out + mlp_out, Shannon + Rényi
```
python workflows/explore_prompt.py
```
### Specify model
```
python workflows/explore_prompt.py --model pythia-1b
```

### Only residual stream hooks
```
python workflows/explore_prompt.py --hooks resid_post resid_pre
```

### Add resid_mid (GPT-2, Gemma-2, Llama only — not Pythia)
```
python workflows/explore_prompt.py --hooks resid_pre resid_mid resid_post attn_out mlp_out
```

### Skip plots, just print summaries
```
python workflows/explore_prompt.py --no-plots
```

### Save entropy records to disk for later multi-model comparison
```
python workflows/explore_prompt.py --save-data --output-dir figures/explore
```


## Basic corpus run commands
```
python workflows/corpus_analysis.py --corpus corpus.json
```
### Different model
```
python workflows/corpus_analysis.py --corpus corpus.json --model pythia-2.8b
```

### Multiple hooks (one forward pass per prompt regardless)
```
python workflows/corpus_analysis.py --corpus corpus.json \
    --hooks resid_post attn_out mlp_out
```
### Filter to one category
```
python workflows/corpus_analysis.py --corpus corpus.json --category pattern
```
### Save data for multi-model comparison later
```
python workflows/corpus_analysis.py --corpus corpus.json \
    --save-data --output-dir figures/corpus
```


## Common Arguments across workflows
| Option | Default |Description |
|--------|-------------|--------|
| `--model` | gpt2-small | Any key from MODEL_CONFIGS in setup.py |
| `--hooks` | resid_post, attn_out, mlp_out | Space Separate hook types |
| `--alpha` | 0.5, 1.0, 2.0, 3.0 | Renyi entropy alpha values |
| `--norm` | energy, abs, softmax | Normalization methods |
| `--output-dir` | figures/explore | Where figures and data land |
| `--no-plots` | off | Skip plotting routines |
| `--save-data` | off | Write .npz for multi-model comparisons |


## Project Structure
```
.
├── data/                   # Saved .npz results (gitignored)
├── figures/                # Generated plots (gitignored)
├── sandbox/                # Deprecated scripts retained for reference
├── workflows/
│   ├── corpus_analysis.py  # Full corpus base/contrast pipeline
│   └── explore_prompt.py   # Single-prompt exploratory analysis
├── computation.py          # Entropy surfaces, EntropyRecord dataclass
├── corpus_gen.py           # Corpus generation
├── entropy_plots.py        # All visualization
├── extraction.py           # Forward pass, ActivationRecord dataclass
└── setup.py                # Model registry, loading, and introspection
```

## Information of hook 
- hook_resid_pre = residual stream before the attention operation at this layer
- hook_attn_out = the output of the attention sub-layer alone (the delta)
- hook_resid_mid = resid_pre + attn_out (residual stream after attention, before MLP)
- hook_mlp_out = the output of the MLP sub-layer alone (the delta)
- hook_resid_post = resid_mid + mlp_out = resid_pre + attn_out + mlp_out
