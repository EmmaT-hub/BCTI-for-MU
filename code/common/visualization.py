
from __future__ import annotations
"""Plot evaluation, tracing, sweep, and training diagnostics."""


from pathlib import Path


import pandas as pd


from config_utils import ensure_dir



def _setup_matplotlib():
    """Configure a noninteractive Matplotlib backend."""
    
    import matplotlib

    
    matplotlib.use("Agg")
    
    import matplotlib.pyplot as plt

    
    return plt



def _save(fig, path: str | Path) -> Path:
    """Save a figure after applying the shared layout settings."""
    
    out = Path(path)
    
    ensure_dir(out.parent)
    
    fig.tight_layout()
    
    fig.savefig(out, dpi=180, bbox_inches="tight")
    
    return out



def plot_target_prob(df: pd.DataFrame, out_path: str | Path, title: str = "Target probability") -> Path:
    """Plot target-answer probabilities by fact and split."""
    
    if df.empty or "first_token_prob" not in df:
        
        return Path(out_path)

    
    plt = _setup_matplotlib()
    
    plot_df = df.copy()
    
    plot_df["label"] = plot_df["id"].astype(str)
    
    colors = plot_df["type"].map({"forget": "#d95f02", "retain": "#1b9e77"}).fillna("#7570b3")

    
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.45), 4.8))
    
    ax.bar(plot_df["label"], plot_df["first_token_prob"], color=colors)
    
    ax.set_title(title)
    
    ax.set_xlabel("fact id")
    
    ax.set_ylabel("first-token probability")
    
    ax.set_ylim(0, max(0.05, min(1.0, float(plot_df["first_token_prob"].max()) * 1.15)))
    
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    
    ax.grid(axis="y", alpha=0.25)
    
    fig.text(0.01, 0.01, "orange=forget, green=retain", fontsize=8)
    
    path = _save(fig, out_path)
    
    plt.close(fig)
    
    return path



def plot_cut_eval(df: pd.DataFrame, out_path: str | Path, title: str = "DPCU-lite cut effect") -> Path:
    """Plot forget and retain metrics before and after cutting."""
    
    required = {"id", "type", "base_first_token_prob", "cut_first_token_prob", "prob_drop"}
    
    if df.empty or not required.issubset(df.columns):
        
        return Path(out_path)

    
    plt = _setup_matplotlib()
    
    plot_df = df.sort_values(["type", "id"]).copy()
    
    x = range(len(plot_df))
    
    width = 0.38
    
    colors = plot_df["type"].map({"forget": "#d95f02", "retain": "#1b9e77"}).fillna("#7570b3")

    
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(plot_df) * 0.55), 4.8))
    
    axes[0].bar([i - width / 2 for i in x], plot_df["base_first_token_prob"], width=width, label="base", color="#9e9e9e")
    
    axes[0].bar([i + width / 2 for i in x], plot_df["cut_first_token_prob"], width=width, label="after cut", color=colors)
    
    axes[0].set_title("Before vs after")
    
    axes[0].set_ylabel("first-token probability")
    
    axes[0].set_xticks(list(x), plot_df["id"].astype(str), rotation=45, ha="right", fontsize=8)
    
    axes[0].legend()
    
    axes[0].grid(axis="y", alpha=0.25)

    
    axes[1].bar(plot_df["id"].astype(str), plot_df["prob_drop"], color=colors)
    
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    
    axes[1].set_title("Probability drop")
    
    axes[1].set_ylabel("base - after cut")
    
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    
    axes[1].grid(axis="y", alpha=0.25)

    
    fig.suptitle(title)
    
    fig.text(0.01, 0.01, "orange=forget, green=retain", fontsize=8)
    
    path = _save(fig, out_path)
    
    plt.close(fig)
    
    return path





def plot_strength_sweep(df: pd.DataFrame, out_path: str | Path, title: str = "Cut strength sweep") -> Path:
    """Plot metrics across candidate cut strengths."""
    
    if df.empty or "cut_strength" not in df:
        
        return Path(out_path)

    
    plt = _setup_matplotlib()
    
    plot_df = df.sort_values("cut_strength").copy()
    
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    
    for col, label, color in [
        
        ("forget_cut_mean", "forget after cut", "#d95f02"),
        
        ("forget_worst_cut_prob", "forget worst after cut", "#a6761d"),
        
        ("forget_tail_cut_prob", "forget hard-tail after cut", "#e6ab02"),
        
        ("forget_drop_mean", "forget drop", "#e7298a"),
        
        ("retain_drop_mean", "retain drop", "#1b9e77"),
        
        ("retain_kl_mean", "retain KL", "#7570b3"),
    
    ]:
        
        if col in plot_df:
            
            ax.plot(plot_df["cut_strength"], plot_df[col], marker="o", label=label, color=color)
    
    if "feasible" in plot_df:
        
        for row in plot_df.itertuples(index=False):
            
            if not bool(getattr(row, "feasible")):
                
                ax.axvspan(float(row.cut_strength) - 0.03, float(row.cut_strength) + 0.03, color="#eeeeee", alpha=0.8)
    
    ax.set_title(title)
    
    ax.set_xlabel("cut strength")
    
    ax.set_ylabel("metric value")
    
    ax.grid(alpha=0.25)
    
    ax.legend()
    
    path = _save(fig, out_path)
    
    plt.close(fig)
    
    return path





def plot_causal_tracing(df: pd.DataFrame, out_path: str | Path, title: str = "Causal tracing restore effect") -> Path:
    """Plot layerwise restoration effects by hook site."""
    
    required = {"layer", "site", "type", "restore_effect"}
    
    if df.empty or not required.issubset(df.columns):
        
        return Path(out_path)

    
    plt = _setup_matplotlib()
    
    grouped = (
        
        df.groupby(["layer", "site", "type"], as_index=False)["restore_effect"]
        
        .mean()
        
        .sort_values(["site", "type", "layer"])
    
    )
    
    sites = list(grouped["site"].drop_duplicates())
    
    fig, axes = plt.subplots(len(sites), 1, figsize=(9, max(3.2, 2.8 * len(sites))), sharex=True)
    
    if len(sites) == 1:
        
        axes = [axes]
    
    for ax, site in zip(axes, sites):
        
        site_df = grouped[grouped["site"] == site]
        
        for fact_type, color in [("forget", "#d95f02"), ("retain", "#1b9e77")]:
            
            type_df = site_df[site_df["type"] == fact_type]
            
            if not type_df.empty:
                
                ax.plot(type_df["layer"], type_df["restore_effect"], marker="o", label=fact_type, color=color)
        
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        
        ax.set_title(site)
        
        ax.set_ylabel("restore effect")
        
        ax.grid(alpha=0.25)
        
        ax.legend()
    
    axes[-1].set_xlabel("layer")
    
    fig.suptitle(title)
    
    path = _save(fig, out_path)
    
    plt.close(fig)
    
    return path



def plot_training_log(df: pd.DataFrame, out_path: str | Path, title: str = "Training loss") -> Path:
    """Plot optimization losses over training steps."""
    
    if df.empty or "step" not in df or "loss" not in df:
        
        return Path(out_path)

    
    plt = _setup_matplotlib()
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    
    ax.plot(df["step"], df["loss"], label="forget loss", color="#d95f02")
    
    if "retain_loss" in df and df["retain_loss"].notna().any():
        
        ax.plot(df["step"], df["retain_loss"], label="retain loss", color="#1b9e77")
    
    ax.set_title(title)
    
    ax.set_xlabel("step")
    
    ax.set_ylabel("loss")
    
    ax.grid(alpha=0.25)
    
    ax.legend()
    
    path = _save(fig, out_path)
    
    plt.close(fig)
    
    return path
