"""dashboard_loader.py — NPZ discovery, loading, caching, and index building.

Discovers .npz files by folder name (folder = record type). Concatenates
multiple files of the same type along axis 0. Provides filtered query methods
that use boolean masking on index arrays — no per-record Python loops in the
hot path.

Record types and their subdirectories:
    entropy      data/entropy/
    wu_subspace  data/wu_subspace/
    ablation     data/ablation/
    ck           data/ck/
    mechanics    data/mechanics/

Usage:
    loader = DashboardLoader("data")
    loader.available_models("entropy")
    result = loader.query_entropy("gpt2-small", "energy", 1.0, "base", "final")
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# Keys that must be present in an entropy/wu_subspace npz to be accepted.
_ENTROPY_REQUIRED_KEYS = {
    "surfaces", "seq_lens", "n_layers", "prompts",
    "model_names", "hook_types", "norm_keys", "alphas",
    "d_models", "roles", "categories",
}
_ABLATION_REQUIRED_KEYS = {
    "kl_divergence", "entropy_change", "top1_preserved",
    "arr_lens", "ks", "ablation_types", "intervention_lyrs",
    "model_names", "prompts", "roles", "categories", "hook_types",
}
_CK_REQUIRED_KEYS = {
    "ck_spectrum", "singular_values", "n_layers", "seq_lens",
    "prompts", "model_names", "hook_types", "d_models",
    "roles", "categories",
}
_MECHANICS_REQUIRED_KEYS = {
    "speed", "acceleration_magnitude", "cosine_sim_state",
    "cosine_sim_update_state", "cosine_sim_update_update",
    "n_trans", "n_accel", "prompts", "model_names", "roles", "categories",
}


def _str_arr(arr: np.ndarray) -> np.ndarray:
    """Ensure object array elements are Python str, not np.str_."""
    return np.array([str(x) for x in arr], dtype=object)


def _pad_to(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """NaN-pad arr to target_shape, which must be >= arr.shape in every dim."""
    if arr.shape == target_shape:
        return arr
    out = np.full(target_shape, np.nan, dtype=arr.dtype)
    slices = tuple(slice(0, s) for s in arr.shape)
    out[slices] = arr
    return out


def _concat_dicts(dicts: list[dict]) -> dict:
    """
    Concatenate a list of key→ndarray dicts along axis 0.

    For 2D+ arrays (surfaces, kl_divergence, etc.) whose trailing dimensions
    differ across files (different max_layers, max_seq_len across models),
    each array is NaN-padded to the global maximum shape before concatenation.
    1D index arrays are concatenated directly.
    """
    if not dicts:
        return {}
    keys = dicts[0].keys()
    out = {}
    for k in keys:
        arrays = [d[k] for d in dicts]
        if arrays[0].ndim == 1:
            out[k] = np.concatenate(arrays, axis=0)
        else:
            # Compute global max shape for trailing dims
            max_shape = tuple(
                max(a.shape[i] for a in arrays)
                for i in range(arrays[0].ndim)
            )
            # Replace axis-0 with per-file size; pad trailing dims
            padded = [_pad_to(a, (a.shape[0],) + max_shape[1:]) for a in arrays]
            out[k] = np.concatenate(padded, axis=0)
    return out


def _load_entropy_file(path: Path) -> dict | None:
    try:
        d = np.load(path, allow_pickle=True)
        if not _ENTROPY_REQUIRED_KEYS.issubset(set(d.keys())):
            return None
        return {
            "surfaces":    d["surfaces"],
            "seq_lens":    d["seq_lens"].astype(np.int32),
            "n_layers":    d["n_layers"].astype(np.int32),
            "model_names": _str_arr(d["model_names"]),
            "hook_types":  _str_arr(d["hook_types"]),
            "norm_keys":   _str_arr(d["norm_keys"]),
            "alphas":      d["alphas"].astype(np.float32),
            "d_models":    d["d_models"].astype(np.int32),
            "roles":       _str_arr(d["roles"]),
            "categories":  _str_arr(d["categories"]),
            "pair_ids":    _str_arr(d["pair_ids"]) if "pair_ids" in d else
                           np.array([""] * len(d["prompts"]), dtype=object),
            "prompts":     _str_arr(d["prompts"]),
        }
    except Exception as e:
        print(f"  [loader] Warning: could not load {path}: {e}")
        return None


def _load_ablation_file(path: Path) -> dict | None:
    try:
        d = np.load(path, allow_pickle=True)
        if not _ABLATION_REQUIRED_KEYS.issubset(set(d.keys())):
            return None
        return {
            "kl_divergence":    d["kl_divergence"].astype(np.float64),
            "entropy_change":   d["entropy_change"].astype(np.float64),
            "top1_preserved":   d["top1_preserved"].astype(bool),
            "arr_lens":         d["arr_lens"].astype(np.int32),
            "ks":               d["ks"].astype(np.int32),
            "ablation_types":   _str_arr(d["ablation_types"]),
            "intervention_lyrs": d["intervention_lyrs"].astype(np.int32),
            "model_names":      _str_arr(d["model_names"]),
            "prompts":          _str_arr(d["prompts"]),
            "roles":            _str_arr(d["roles"]),
            "categories":       _str_arr(d["categories"]),
            "hook_types":       _str_arr(d["hook_types"]),
        }
    except Exception as e:
        print(f"  [loader] Warning: could not load {path}: {e}")
        return None


def _load_mechanics_file(path: Path) -> dict | None:
    try:
        d = np.load(path, allow_pickle=True)
        if not _MECHANICS_REQUIRED_KEYS.issubset(set(d.keys())):
            return None
        return {
            "speed":                    d["speed"].astype(np.float64),
            "acceleration_magnitude":   d["acceleration_magnitude"].astype(np.float64),
            "cosine_sim_state":         d["cosine_sim_state"].astype(np.float64),
            "cosine_sim_update_state":  d["cosine_sim_update_state"].astype(np.float64),
            "cosine_sim_update_update": d["cosine_sim_update_update"].astype(np.float64),
            "n_trans":      d["n_trans"].astype(np.int32),
            "n_accel":      d["n_accel"].astype(np.int32),
            "prompts":      _str_arr(d["prompts"]),
            "model_names":  _str_arr(d["model_names"]),
            "roles":        _str_arr(d["roles"]),
            "categories":   _str_arr(d["categories"]),
            "pair_ids":     _str_arr(d["pair_ids"]) if "pair_ids" in d else
                            np.array([""] * len(d["prompts"]), dtype=object),
        }
    except Exception as e:
        print(f"  [loader] Warning: could not load {path}: {e}")
        return None


def _load_ck_file(path: Path) -> dict | None:
    try:
        d = np.load(path, allow_pickle=True)
        if not _CK_REQUIRED_KEYS.issubset(set(d.keys())):
            return None
        return {
            "ck_spectrum":     d["ck_spectrum"].astype(np.float32),
            "singular_values": d["singular_values"].astype(np.float32),
            "n_layers":        d["n_layers"].astype(np.int32),
            "seq_lens":        d["seq_lens"].astype(np.int32),
            "prompts":         _str_arr(d["prompts"]),
            "model_names":     _str_arr(d["model_names"]),
            "hook_types":      _str_arr(d["hook_types"]),
            "d_models":        d["d_models"].astype(np.int32),
            "roles":           _str_arr(d["roles"]),
            "categories":      _str_arr(d["categories"]),
            "pair_ids":        _str_arr(d["pair_ids"]) if "pair_ids" in d else
                               np.array([""] * len(d["prompts"]), dtype=object),
        }
    except Exception as e:
        print(f"  [loader] Warning: could not load {path}: {e}")
        return None


class DashboardLoader:
    """
    Loads and indexes all .npz files from a structured data directory.

    On construction, all files are loaded eagerly and concatenated by type.
    All subsequent queries are pure numpy boolean masking — no file I/O.
    """

    _SUBDIRS = {
        "entropy":     ("entropy",     _load_entropy_file),
        "wu_subspace": ("wu_subspace", _load_entropy_file),   # same schema
        "ablation":    ("ablation",    _load_ablation_file),
        "ck":          ("ck",          _load_ck_file),
        "mechanics":   ("mechanics",   _load_mechanics_file),
    }

    def __init__(self, data_root: str = "data"):
        self._root = Path(data_root)
        self._index: dict[str, dict] = {}
        self._load_all()

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        for record_type, (subdir, loader_fn) in self._SUBDIRS.items():
            folder = self._root / subdir
            if not folder.exists():
                print(f"  [loader] {record_type}: directory not found ({folder})")
                self._index[record_type] = {}
                continue

            file_dicts = []
            for path in sorted(folder.glob("*.npz")):
                d = loader_fn(path)
                if d is not None:
                    file_dicts.append(d)
                    print(f"  [loader] loaded {path.name} ({len(d['prompts'])} records)")

            if file_dicts:
                self._index[record_type] = _concat_dicts(file_dicts)
                n = len(self._index[record_type]["prompts"])
                print(f"  [loader] {record_type}: {n} total records from {len(file_dicts)} file(s)")
            else:
                self._index[record_type] = {}
                print(f"  [loader] {record_type}: no valid files found")

    def _idx(self, record_type: str) -> dict:
        return self._index.get(record_type, {})

    def _has(self, record_type: str) -> bool:
        return bool(self._index.get(record_type))

    # ── Discovery ──────────────────────────────────────────────────────────────

    def available_models(self, record_type: str) -> list[str]:
        d = self._idx(record_type)
        if not d:
            return []
        return sorted(set(str(x) for x in d["model_names"]))

    def available_norm_keys(self, record_type: str, model: str) -> list[str]:
        d = self._idx(record_type)
        if not d:
            return []
        mask = d["model_names"] == model
        return sorted(set(str(x) for x in d["norm_keys"][mask]))

    def available_alphas(self, record_type: str, model: str) -> list[float]:
        d = self._idx(record_type)
        if not d:
            return []
        mask = d["model_names"] == model
        return sorted(set(float(x) for x in d["alphas"][mask]))

    def available_roles(self, record_type: str, model: str) -> list[str]:
        d = self._idx(record_type)
        if not d:
            return []
        mask = d["model_names"] == model
        return sorted(set(str(x) for x in d["roles"][mask]))

    def available_k_values(self, record_type: str, model: str) -> list[int]:
        """Ablation only — returns sorted list of subspace ranks k."""
        d = self._idx(record_type)
        if not d or "ks" not in d:
            return []
        mask = d["model_names"] == model
        return sorted(set(int(x) for x in d["ks"][mask]))

    def available_wu_directions(self, model: str) -> list[str]:
        """WU subspace — returns ['orthogonal', 'parallel'] if both present."""
        nk_list = self.available_norm_keys("wu_subspace", model)
        dirs = set()
        for nk in nk_list:
            m = re.match(r"wu_(parallel|orthogonal)_k\d+", nk)
            if m:
                dirs.add(m.group(1))
        return sorted(dirs)

    def available_wu_k_values(self, model: str, direction: str) -> list[int]:
        """WU subspace — rank values available for a given direction."""
        nk_list = self.available_norm_keys("wu_subspace", model)
        ks = set()
        prefix = f"wu_{direction}_k"
        for nk in nk_list:
            if nk.startswith(prefix):
                try:
                    ks.add(int(nk[len(prefix):]))
                except ValueError:
                    pass
        return sorted(ks)

    def available_ck_models(self) -> list[str]:
        return self.available_models("ck")

    def max_n_layers(self, record_type: str, model: str) -> int:
        """Maximum n_layers for a model in the given record type."""
        d = self._idx(record_type)
        if not d or "n_layers" not in d:
            return 12
        mask = d["model_names"] == model
        if not mask.any():
            return 12
        return int(d["n_layers"][mask].max())

    # ── Query: entropy / WU subspace (shared schema) ──────────────────────────

    def _query_surface(
        self,
        record_type: str,
        model: str,
        norm_key: str,
        alpha: float,
        role: str,
        position_mode: str = "final",
    ) -> dict:
        """
        Core query for entropy and wu_subspace records.

        Returns
        -------
        dict with keys:
            "surfaces"   : list of 1D arrays [n_layers_i], one per prompt
            "categories" : object array [n_prompts]
            "pair_ids"   : object array [n_prompts]
            "model"      : str
        """
        d = self._idx(record_type)
        if not d:
            return {"surfaces": [], "categories": np.array([]),
                    "pair_ids": np.array([]), "model": model}

        mask = (
            (d["model_names"] == model) &
            (d["norm_keys"]   == norm_key) &
            (np.abs(d["alphas"] - float(alpha)) < 1e-5) &
            (d["roles"]       == role)
        )
        indices = np.where(mask)[0]

        profiles = []
        for i in indices:
            nl = int(d["n_layers"][i])
            sl = int(d["seq_lens"][i])
            surf = d["surfaces"][i, :nl, :sl]   # [n_layers, seq_len]
            if position_mode == "final":
                profiles.append(surf[:, -1].copy())
            else:  # "mean"
                profiles.append(surf.mean(axis=1).copy())

        return {
            "surfaces":   profiles,
            "categories": d["categories"][indices],
            "pair_ids":   d["pair_ids"][indices],
            "model":      model,
        }

    def query_entropy(
        self,
        model: str,
        norm_key: str,
        alpha: float,
        role: str,
        position_mode: str = "final",
    ) -> dict:
        return self._query_surface("entropy", model, norm_key, alpha,
                                   role, position_mode)

    def query_wu_subspace(
        self,
        model: str,
        direction: str,
        k: int,
        alpha: float,
        role: str,
        position_mode: str = "final",
    ) -> dict:
        norm_key = f"wu_{direction}_k{k}"
        return self._query_surface("wu_subspace", model, norm_key, alpha,
                                   role, position_mode)

    # ── Query: ablation ────────────────────────────────────────────────────────

    def query_ablation_posthoc(
        self,
        model: str,
        k: int,
        role: str,
    ) -> dict:
        """
        Returns posthoc ablation profiles for a given (model, k, role).

        Returns
        -------
        dict with keys:
            "kl"             : list of 1D arrays [n_layers_i]
            "top1"           : list of 1D arrays [n_layers_i] (float 0/1)
            "entropy_change" : list of 1D arrays [n_layers_i]
            "categories"     : object array
        """
        d = self._idx("ablation")
        if not d:
            return {"kl": [], "top1": [], "entropy_change": [], "categories": np.array([])}

        mask = (
            (d["model_names"]    == model) &
            (d["ks"]             == int(k)) &
            (d["ablation_types"] == "posthoc") &
            (d["roles"]          == role)
        )
        indices = np.where(mask)[0]

        kl_list  = []
        t1_list  = []
        ec_list  = []
        for i in indices:
            nl = int(d["arr_lens"][i])
            kl_list.append(d["kl_divergence"][i,  :nl].copy())
            t1_list.append(d["top1_preserved"][i, :nl].astype(float).copy())
            ec_list.append(d["entropy_change"][i, :nl].copy())

        return {
            "kl":             kl_list,
            "top1":           t1_list,
            "entropy_change": ec_list,
            "categories":     d["categories"][indices],
        }

    def query_ablation_intervention(
        self,
        model: str,
        role: str,
    ) -> dict:
        """
        Build (n_k × n_layers) top1_preserved heatmap matrices for base and contrast.

        Returns
        -------
        dict with keys:
            "top1_base"     : ndarray [n_k, n_layers] — mean top1 preservation
            "top1_contrast" : ndarray [n_k, n_layers]
            "k_values"      : list of int (rows)
            "layer_indices" : list of int (columns, 0-based)
        """
        d = self._idx("ablation")
        if not d:
            return {"top1_base": np.zeros((0, 0)), "top1_contrast": np.zeros((0, 0)),
                    "k_values": [], "layer_indices": []}

        # Determine axes from data
        model_mask = (d["model_names"] == model) & (d["ablation_types"] == "intervention")
        if not model_mask.any():
            return {"top1_base": np.zeros((0, 0)), "top1_contrast": np.zeros((0, 0)),
                    "k_values": [], "layer_indices": []}

        k_values      = sorted(set(int(x) for x in d["ks"][model_mask]))
        layer_indices = sorted(set(int(x) for x in d["intervention_lyrs"][model_mask]
                                   if int(x) >= 0))

        def _build_matrix(r: str) -> np.ndarray:
            mat = np.full((len(k_values), len(layer_indices)), np.nan)
            for ki, k in enumerate(k_values):
                for li, lyr in enumerate(layer_indices):
                    cell_mask = (
                        model_mask &
                        (d["ks"]               == k) &
                        (d["intervention_lyrs"] == lyr) &
                        (d["roles"]            == r)
                    )
                    idx = np.where(cell_mask)[0]
                    if len(idx) == 0:
                        continue
                    # intervention arr_lens is 1 — top1_preserved shape [N, 1]
                    vals = d["top1_preserved"][idx, 0].astype(float)
                    mat[ki, li] = vals.mean()
            return mat

        return {
            "top1_base":     _build_matrix("base"),
            "top1_contrast": _build_matrix("contrast"),
            "k_values":      k_values,
            "layer_indices": layer_indices,
        }

    # ── Query: mechanics ──────────────────────────────────────────────────────

    def query_mechanics(self, model: str, role: str) -> dict:
        """
        Return all five mechanical scalar curves for a given (model, role).

        Each array in the returned lists is trimmed to its true length using
        n_trans / n_accel so callers never see NaN padding.

        Returns
        -------
        dict with keys:
            "speed"                    : list of 1D arrays [n_trans_i]
            "acceleration_magnitude"   : list of 1D arrays [n_accel_i]
            "cosine_sim_state"         : list of 1D arrays [n_trans_i]
            "cosine_sim_update_state"  : list of 1D arrays [n_trans_i]
            "cosine_sim_update_update" : list of 1D arrays [n_accel_i]
            "categories"               : object array
        """
        d = self._idx("mechanics")
        empty = {
            "speed": [], "acceleration_magnitude": [],
            "cosine_sim_state": [], "cosine_sim_update_state": [],
            "cosine_sim_update_update": [], "categories": np.array([]),
        }
        if not d:
            return empty

        mask    = (d["model_names"] == model) & (d["roles"] == role)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return empty

        def _trim_trans(i):
            nt = int(d["n_trans"][i])
            return {
                "speed":                   d["speed"][i, :nt].copy(),
                "cosine_sim_state":        d["cosine_sim_state"][i, :nt].copy(),
                "cosine_sim_update_state": d["cosine_sim_update_state"][i, :nt].copy(),
            }

        def _trim_accel(i):
            na = int(d["n_accel"][i])
            return {
                "acceleration_magnitude":   d["acceleration_magnitude"][i, :na].copy(),
                "cosine_sim_update_update": d["cosine_sim_update_update"][i, :na].copy(),
            }

        out = {k: [] for k in empty if k != "categories"}
        for i in indices:
            t = _trim_trans(i)
            a = _trim_accel(i)
            out["speed"].append(t["speed"])
            out["cosine_sim_state"].append(t["cosine_sim_state"])
            out["cosine_sim_update_state"].append(t["cosine_sim_update_state"])
            out["acceleration_magnitude"].append(a["acceleration_magnitude"])
            out["cosine_sim_update_update"].append(a["cosine_sim_update_update"])

        out["categories"] = d["categories"][indices]
        return out

    def available_mechanics_models(self) -> list[str]:
        return self.available_models("mechanics")

    # ── Query: c_k spectrum ────────────────────────────────────────────────────

    def query_ck(
        self,
        model: str,
        role: str,
        layer_idx: int,
        position_mode: str = "final",
    ) -> dict:
        """
        Returns mean ± SEM c_k spectrum at a specific layer across prompts.

        Returns
        -------
        dict with keys:
            "ck_mean"        : 1D array [d_model] or None
            "ck_sem"         : 1D array [d_model] or None
            "singular_values": 1D array [d_model] or None
            "categories"     : object array
        """
        d = self._idx("ck")
        empty = {"ck_mean": None, "ck_sem": None,
                 "singular_values": None, "categories": np.array([])}
        if not d:
            return empty

        mask = (d["model_names"] == model) & (d["roles"] == role)
        # also filter to records that have the requested layer
        mask &= d["n_layers"] > layer_idx
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return empty

        ck_vecs = []
        for i in indices:
            nl = int(d["n_layers"][i])
            sl = int(d["seq_lens"][i])
            spec = d["ck_spectrum"][i, :nl, :sl, :]   # [n_layers, seq_len, d_model]
            layer_spec = spec[layer_idx]               # [seq_len, d_model]
            if position_mode == "final":
                ck_vecs.append(layer_spec[-1])
            else:
                ck_vecs.append(layer_spec.mean(axis=0))

        arr = np.array(ck_vecs, dtype=np.float32)       # [n_prompts, d_model]
        ck_mean = arr.mean(axis=0)
        ck_sem  = arr.std(axis=0, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(ck_mean)

        sv = d["singular_values"][indices[0]]            # same for all records same model

        return {
            "ck_mean":         ck_mean,
            "ck_sem":          ck_sem,
            "singular_values": sv,
            "categories":      d["categories"][indices],
        }
