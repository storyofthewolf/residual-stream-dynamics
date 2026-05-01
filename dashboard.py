"""dashboard.py — Gradio analysis dashboard for residual-stream-dynamics.

Tabs:
    1. Entropy          — entropy surface curves, base vs contrast
    2. WU Subspace      — W_U projection entropy (r‖ and r⊥)
    3. Ablation         — posthoc curves and intervention heatmap
    4. C_k Spectra      — c_k spectrum at a selected layer

No computation or figure construction lives here.
Callbacks do exactly: call loader → call viz → return figure.

Usage:
    python dashboard.py --data-root data/
    python dashboard.py --data-root data/ --share
"""

import argparse

import gradio as gr

import dashboard_loader as loader_mod
import dashboard_viz    as viz

# Module-level loader, initialized in main() before demo.launch()
loader: loader_mod.DashboardLoader | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _first(lst: list, fallback=None):
    return lst[0] if lst else fallback


def _alpha_choices(alphas: list[float]) -> list[str]:
    return [str(a) for a in alphas]


def _alpha_label(a: float) -> str:
    return "1.0 (Shannon)" if abs(a - 1.0) < 1e-4 else str(a)


# ── Tab 1: Entropy ─────────────────────────────────────────────────────────────

def _entropy_models():
    return loader.available_models("entropy")

def update_entropy_norm_keys(model):
    nk = loader.available_norm_keys("entropy", model)
    return gr.update(choices=nk, value=_first(nk))

def update_entropy_alphas(model):
    al = loader.available_alphas("entropy", model)
    ch = _alpha_choices(al)
    return gr.update(choices=ch, value=_first(ch))

def update_entropy_plot(model, norm_key, alpha_str, position_mode):
    alpha = float(alpha_str)
    base     = loader.query_entropy(model, norm_key, alpha, "base",     position_mode)
    contrast = loader.query_entropy(model, norm_key, alpha, "contrast", position_mode)
    return viz.plot_entropy_curves(
        base["surfaces"], contrast["surfaces"],
        model, norm_key, alpha, position_mode,
    )


# ── Tab 2: WU Subspace ─────────────────────────────────────────────────────────

def _wu_models():
    return loader.available_models("wu_subspace")

def update_wu_k_values(model, direction):
    ks = loader.available_wu_k_values(model, direction)
    ch = [str(k) for k in ks]
    return gr.update(choices=ch, value=_first(ch))

def update_wu_alphas(model):
    al = loader.available_alphas("wu_subspace", model)
    ch = _alpha_choices(al)
    return gr.update(choices=ch, value=_first(ch))

def update_wu_plot(model, direction, k_str, alpha_str, position_mode):
    k     = int(k_str)
    alpha = float(alpha_str)
    base     = loader.query_wu_subspace(model, direction, k, alpha, "base",     position_mode)
    contrast = loader.query_wu_subspace(model, direction, k, alpha, "contrast", position_mode)
    return viz.plot_wu_subspace_curves(
        base["surfaces"], contrast["surfaces"],
        model, direction, k, alpha,
    )


# ── Tab 3: Ablation ────────────────────────────────────────────────────────────

def _ablation_models():
    return loader.available_models("ablation")

def update_ablation_k_values(model):
    ks = loader.available_k_values("ablation", model)
    ch = [str(k) for k in ks]
    return gr.update(choices=ch, value=_first(ch))

def update_posthoc_plot(model, k_str):
    k        = int(k_str)
    base     = loader.query_ablation_posthoc(model, k, "base")
    contrast = loader.query_ablation_posthoc(model, k, "contrast")
    return viz.plot_ablation_posthoc(
        base["top1"],     contrast["top1"],
        base["kl"],       contrast["kl"],
        model, k,
    )

def update_intervention_plot(model):
    result = loader.query_ablation_intervention(model, role="base")
    result_c = loader.query_ablation_intervention(model, role="contrast")
    # query_ablation_intervention already builds combined matrices — merge them
    top1_base     = result["top1_base"]
    top1_contrast = result_c["top1_contrast"]
    k_values      = result["k_values"]
    layer_indices = result["layer_indices"]
    return viz.plot_ablation_heatmap(top1_base, top1_contrast, k_values, layer_indices, model)


# ── Tab 4: C_k Spectra ─────────────────────────────────────────────────────────

def _ck_models():
    return loader.available_ck_models()

def update_ck_layer_slider(model):
    n = loader.max_n_layers("ck", model) if model else 12
    return gr.update(maximum=max(n - 1, 0), value=min(6, max(n - 1, 0)))

def update_ck_plot(model, role, layer_idx, position_mode):
    if not model:
        return viz.plot_ck_spectrum(None, None, None, "—", 0, role)
    result = loader.query_ck(model, role, int(layer_idx), position_mode)
    return viz.plot_ck_spectrum(
        result["ck_mean"], result["ck_sem"], result["singular_values"],
        model, int(layer_idx), role,
    )


# ── App layout ─────────────────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    entropy_models   = _entropy_models()
    wu_models        = _wu_models()
    ablation_models  = _ablation_models()
    ck_models        = _ck_models()

    def _e_nk(m):  return loader.available_norm_keys("entropy", m) if m else []
    def _e_al(m):  return _alpha_choices(loader.available_alphas("entropy", m)) if m else []
    def _wu_k(m, d): return [str(k) for k in loader.available_wu_k_values(m, d)] if m else []
    def _wu_al(m): return _alpha_choices(loader.available_alphas("wu_subspace", m)) if m else []
    def _ab_k(m):  return [str(k) for k in loader.available_k_values("ablation", m)] if m else []

    e_m0  = _first(entropy_models)
    wu_m0 = _first(wu_models)
    ab_m0 = _first(ablation_models)
    ck_m0 = _first(ck_models)

    with gr.Blocks(title="Residual Stream Dynamics") as demo:
        gr.Markdown("# Residual Stream Dynamics — Analysis Dashboard")
        gr.Markdown(
            "Interactive visualization of entropy surfaces, WU subspace projections, "
            "ablation results, and c_k spectra. All data is precomputed — no model inference."
        )

        with gr.Tabs():

            # ── Tab 1: Entropy ─────────────────────────────────────────────
            with gr.Tab("Entropy"):
                with gr.Row():
                    with gr.Column(scale=1):
                        e_model = gr.Dropdown(
                            choices=entropy_models,
                            value=e_m0,
                            label="Model",
                        )
                        e_norm_key = gr.Dropdown(
                            choices=_e_nk(e_m0),
                            value=_first(_e_nk(e_m0)),
                            label="Norm key",
                        )
                        e_alpha = gr.Dropdown(
                            choices=_e_al(e_m0),
                            value=_first(_e_al(e_m0)),
                            label="Rényi α",
                        )
                        e_position = gr.Radio(
                            choices=["final", "mean"],
                            value="final",
                            label="Token position",
                        )
                        e_btn = gr.Button("Update Plot", variant="primary")
                    with gr.Column(scale=3):
                        e_plot = gr.Plot(label="Entropy curves")

                # Chained updates: model → norm_key choices, alpha choices
                e_model.change(update_entropy_norm_keys, inputs=e_model, outputs=e_norm_key)
                e_model.change(update_entropy_alphas,    inputs=e_model, outputs=e_alpha)

                e_btn.click(
                    update_entropy_plot,
                    inputs=[e_model, e_norm_key, e_alpha, e_position],
                    outputs=e_plot,
                )

            # ── Tab 2: WU Subspace ─────────────────────────────────────────
            with gr.Tab("WU Subspace"):
                wu_dir0  = "parallel"
                wu_k0    = _first(_wu_k(wu_m0, wu_dir0))
                wu_al0   = _first(_wu_al(wu_m0))
                with gr.Row():
                    with gr.Column(scale=1):
                        wu_model = gr.Dropdown(
                            choices=wu_models,
                            value=wu_m0,
                            label="Model",
                        )
                        wu_direction = gr.Radio(
                            choices=["parallel", "orthogonal"],
                            value="parallel",
                            label="Subspace direction  (r‖ / r⊥)",
                        )
                        wu_k = gr.Dropdown(
                            choices=_wu_k(wu_m0, wu_dir0),
                            value=wu_k0,
                            label="Subspace rank k",
                        )
                        wu_alpha = gr.Dropdown(
                            choices=_wu_al(wu_m0),
                            value=wu_al0,
                            label="Rényi α",
                        )
                        wu_position = gr.Radio(
                            choices=["final", "mean"],
                            value="final",
                            label="Token position",
                        )
                        wu_btn = gr.Button("Update Plot", variant="primary")
                    with gr.Column(scale=3):
                        wu_plot = gr.Plot(label="WU subspace entropy curves")

                wu_model.change(update_wu_alphas, inputs=wu_model, outputs=wu_alpha)
                wu_model.change(
                    update_wu_k_values,
                    inputs=[wu_model, wu_direction],
                    outputs=wu_k,
                )
                wu_direction.change(
                    update_wu_k_values,
                    inputs=[wu_model, wu_direction],
                    outputs=wu_k,
                )

                wu_btn.click(
                    update_wu_plot,
                    inputs=[wu_model, wu_direction, wu_k, wu_alpha, wu_position],
                    outputs=wu_plot,
                )

            # ── Tab 3: Ablation ────────────────────────────────────────────
            with gr.Tab("Ablation"):
                ab_k0 = _first(_ab_k(ab_m0))
                with gr.Tabs():

                    with gr.Tab("Posthoc"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                ab_ph_model = gr.Dropdown(
                                    choices=ablation_models,
                                    value=ab_m0,
                                    label="Model",
                                )
                                ab_ph_k = gr.Dropdown(
                                    choices=_ab_k(ab_m0),
                                    value=ab_k0,
                                    label="Subspace rank k",
                                )
                                ab_ph_btn = gr.Button("Update Plot", variant="primary")
                            with gr.Column(scale=3):
                                ab_ph_plot = gr.Plot(label="Posthoc ablation")

                        ab_ph_model.change(
                            update_ablation_k_values, inputs=ab_ph_model, outputs=ab_ph_k
                        )
                        ab_ph_btn.click(
                            update_posthoc_plot,
                            inputs=[ab_ph_model, ab_ph_k],
                            outputs=ab_ph_plot,
                        )

                    with gr.Tab("Intervention Heatmap"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                ab_int_model = gr.Dropdown(
                                    choices=ablation_models,
                                    value=ab_m0,
                                    label="Model",
                                )
                                ab_int_btn = gr.Button("Update Plot", variant="primary")
                            with gr.Column(scale=3):
                                ab_int_plot = gr.Plot(label="Intervention heatmap")

                        ab_int_btn.click(
                            update_intervention_plot,
                            inputs=ab_int_model,
                            outputs=ab_int_plot,
                        )

            # ── Tab 4: C_k Spectra ─────────────────────────────────────────
            with gr.Tab("C_k Spectra"):
                ck_n0 = loader.max_n_layers("ck", ck_m0) if ck_m0 else 12
                ck_layer_max = max(ck_n0 - 1, 0)
                with gr.Row():
                    with gr.Column(scale=1):
                        if ck_models:
                            ck_model = gr.Dropdown(
                                choices=ck_models,
                                value=ck_m0,
                                label="Model",
                            )
                            ck_role = gr.Radio(
                                choices=["base", "contrast"],
                                value="base",
                                label="Role",
                            )
                            ck_layer = gr.Slider(
                                minimum=0,
                                maximum=ck_layer_max,
                                step=1,
                                value=min(6, ck_layer_max),
                                label="Layer index",
                            )
                            ck_position = gr.Radio(
                                choices=["final", "mean"],
                                value="final",
                                label="Token position",
                            )
                            ck_btn = gr.Button("Update Plot", variant="primary")
                        else:
                            gr.Markdown(
                                "**No c_k data found.**\n\n"
                                "Generate c_k records with:\n"
                                "```\npython workflows/ck_analysis.py "
                                "--corpus <corpus.json> --save-data\n```"
                            )
                    with gr.Column(scale=3):
                        ck_plot = gr.Plot(label="c_k spectrum")

                if ck_models:
                    ck_model.change(
                        update_ck_layer_slider, inputs=ck_model, outputs=ck_layer
                    )
                    ck_btn.click(
                        update_ck_plot,
                        inputs=[ck_model, ck_role, ck_layer, ck_position],
                        outputs=ck_plot,
                    )

    return demo


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Residual Stream Dynamics analysis dashboard"
    )
    parser.add_argument("--data-root", default="data",
                        help="Root directory containing entropy/, wu_subspace/, "
                             "ablation/, ck/ subdirectories")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    print(f"\nLoading data from '{args.data_root}'...")
    loader = loader_mod.DashboardLoader(args.data_root)

    demo = build_demo()
    demo.queue()
    demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Base())
