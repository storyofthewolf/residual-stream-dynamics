
## Technical 

# redundancy of SVD utilities
def wu_explained_variance()
def compute_wu_svd()

These are duplicated between ablation_compute.py and entropy_compute.py, with only subtle differences in argument structure.
Versions in ablation_compute.py are called in workflows/ablation_analysis.py
Versions in entropy_compute.py are called in workflows/wu_subspace_analysis.py

 These could be unified at a later date into a svd_utls.py file.  Leave for now.


## Science Direction

# Von Neumann entropy calculation
# Prompt entropy
# Renyi spectrum and multifractal analysis


# Quantify the influence of dimension k of the logit distribution
- projection of r onto the -k-th right singular vector, weighted by sigma_k
- For each prompt and each layer, compute:
effective influence spectrum: c_k = σ_k · (r · v_k)
- entropy of the weighted projections

A direct test of this hypothesis would compute the per-direction logit influence spectrum c_k = σ_k(r · v_k) and compare its distribution across prompt types." That signals you know exactly what the next experiment is.

