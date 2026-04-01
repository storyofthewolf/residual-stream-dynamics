# residual stream dynamics

A package for analysising the residual stream and related layer properties of toy LLMs

## File Structure
- extraction.py — defines ActivationRecord and extract_activations(). The critical function accepts a list of hook types and returns a dict of records from one forward pass. The HOOK_TYPES registry at the top is where you add new hook types — no other changes needed elsewhere. The has_resid_mid flag is detected automatically by probing the cache before the main forward pass.
- entropy_compute.py — defines EntropyRecord and compute_entropy_surface(). One ActivationRecord → multiple EntropyRecords (one per norm×alpha combination). The filter_records() helper is the primary tool for selecting subsets downstream. Future analysis types (RenyiSpectrumRecord, VonNeumannRecord, DynamicsRecord) follow the same pattern — new dataclass, new compute function, same ActivationRecord input.
- entropy_plots.py — all plot functions now accept EntropyRecord objects directly. Hook type, norm, alpha, d_model, and prompt are all read from the record rather than passed as separate arguments. The new plot_hook_comparison() is the primary tool for your attention vs MLP scientific question.
- ablation_compute.py - defines AblationRecord and functions used in posthoc ablation and interventonal ablation experiments.
- ablation_plots.py - all plot function accept AblationRecord objects, performs various plotting routines.
- corpus_gen.py - generates corpus of prompts
- workflows/single_prompt.py — edit DEFAULT_PROMPTS at the top for quick experiments. Runs --hooks resid_post attn_out mlp_out by default, generating the hook comparison plot automatically. Use --save-data to write results for later multi-model comparison.
- workflows/entropy_analysis.py — drops in as a replacement for the old corpus mode in residual_stream_entropy.py. Same --corpus and --model flags as before.
- workflows/wu_subspace_analysis.py - workflow for residual stream SVD 
- workflows/ablation_analysis.py - workflor for ablation experiments


## Quick Start Commands for single prompt
### Default: gpt2-small, hooks resid_post + attn_out + mlp_out, Shannon + Rényi
```
python workflows/single_prompt.py
```
### Specify model
```
python workflows/single_prompt.py --model pythia-1b
```

### Only residual stream hooks
```
python workflows/single_prompt.py --hooks resid_post resid_pre
```

### Add resid_mid (GPT-2, Gemma-2, Llama only — not Pythia)
```
python workflows/single_prompt.py --hooks resid_pre resid_mid resid_post attn_out mlp_out
```

### Skip plots, just print summaries
```
python workflows/single_prompt.py --no-plots
```

### Save entropy records to disk for later multi-model comparison
```
python workflows/single_prompt.py --save-data --output-dir figures/explore
```


## Quick Start commands for corpus run
```
python workflows/entropy_analysis.py --corpus corpus.json
```
### Different model
```
python workflows/entropy_analysis.py --corpus corpus.json --model pythia-2.8b
```

### Multiple hooks (one forward pass per prompt regardless)
```
python workflows/entropy_analysis.py --corpus corpus.json \
    --hooks resid_post attn_out mlp_out
```
### Filter to one category
```
python workflows/entropy_analysis.py --corpus corpus.json --category pattern
```
### Save data for multi-model comparison later
```
python workflows/entropy_analysis.py --corpus corpus.json \
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

## See specific workflow scripts for unique arguments

## Project Structure
```
.
├── data/                        # Saved .npz results (gitignored)
├── figures/                     # Generated plots (gitignored)
├── sandbox/                     # Deprecated scripts retained for reference
├── workflows/
    ├── entropy_analysis.py      # Full corpus base/contrast pipeline
    ├── ablation_analysis.py
    ├── wu_subspace_analysis.py
    └── single_prompt.py         # Single-prompt exploratory analysis
├── entropy_compute.py           # EntropyRecord dataclass and entropy calc functions
├── entropy_plots.py             # Entropy multiplot visualization
├── ablation_compute.py          # AblationRecord dataclass and ablation calc functions 
├── ablation_plots.py            # Ablation multiplot visualization 
├── extraction.py                # ActivationRecord dataclass and forward pass
├── corpus_gen.py                # Corpus generation
└── setup.py                     # Model registry, loading, and introspection

```

## Information of hook 
- hook_resid_pre = residual stream before the attention operation at this layer
- hook_attn_out = the output of the attention sub-layer alone (the delta)
- hook_resid_mid = resid_pre + attn_out (residual stream after attention, before MLP)
- hook_mlp_out = the output of the MLP sub-layer alone (the delta)
- hook_resid_post = resid_mid + mlp_out = resid_pre + attn_out + mlp_out
