# Future Work
April 1st, 2026

# Science

## Von Neumann entropy calculation
## Prompt entropy
## Renyi spectrum and multifractal analysis


## Quantify the influence of dimension k of the logit distribution
- projection of r onto the -k-th right singular vector, weighted by sigma_k
- For each prompt and each layer, compute:
effective influence spectrum: c_k = σ_k · (r · v_k)
- entropy of the weighted projections

A direct test of this hypothesis would compute the per-direction logit influence spectrum c_k = σ_k(r · v_k) and compare its distribution across prompt types." That signals you know exactly what the next experiment is.

## tests randomization of content from residual stream matrix.


# Technical 

## redundancy of SVD utilities
def wu_explained_variance()
def compute_wu_svd()

These are duplicated between ablation_compute.py and entropy_compute.py, with only subtle differences in argument structure.
Versions in ablation_compute.py are called in workflows/ablation_analysis.py
Versions in entropy_compute.py are called in workflows/wu_subspace_analysis.py

 These could be unified at a later date into a svd_utls.py file.  Leave for now.

## inconsistency in argument passing of filenames between entropy_plots.py and ablation_plots.py

entropy_plots.py funcations takes
    output_dir: Path,
    filename:   Optional[str] = None
then merges the file names within each function

ablation_plots.py functions takes
    save_path:  str
which has a merged path+filename passed from the workflow scripts

## institute more robust data and plot naming conventions if parameter sweeps significantly proliferate
- plotting functions in  entropy_plots.py have different arguments lists for those driven for corpus vs single prompt testing
