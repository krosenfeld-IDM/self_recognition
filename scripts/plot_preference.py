"""
pip install polars seaborn
"""
import json
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="ticks", palette="muted")

root_dir = Path(__file__).parents[1]

def analyze_pairwise_preference(dataset_filter = None, suffix:str = None):
    """ 
    I am assuming that my results come from experiments.py
    """
    datasets = ['cnn', 'xsum']
    models = ['llama', 'gpt35', 'gpt4']

    # construct the dataframe
    df = None
    for dataset in datasets:
        for model in models:
            # load the result data
            df_ = pl.read_json(root_dir / 'results' / dataset / f'{model}_results.json')
            # update the "model" column to "smodel" (the model providing the summary)
            df_ = df_.rename({'model': 'smodel'})
            df_ = df_.with_columns(model = pl.lit(model))
            df_ = df_.with_columns(dataset = pl.lit(dataset))
            keep_cols = ['model', 'smodel', 'dataset', 'self_preference']
            if df is None:
                df = df_[keep_cols]
            else:
                df = pl.concat((df, df_[keep_cols]))


    if dataset_filter is not None:
        df = df.filter(pl.col('dataset').is_in(dataset_filter))

    df = df.rename({'self_preference': 'score'})
    num_models = df['model'].n_unique()
    all_models = ['llama', 'gpt35', 'gpt4', 'claude', 'human']

    fig, axes = plt.subplots(1, num_models, figsize=(10, 5))
    for i, model in enumerate(models):
        ax = axes[i]
        group = df.filter(pl.col('model') == model)
        valid_models = [m for m in all_models if m in group['smodel'].unique()]
        order = valid_models

        # Draw boxplot (distribution)
        sns.boxplot(
            data=group,
            x="smodel",
            y="score",
            showfliers=False,
            ax=ax,
            legend=False,
            order=order,
        )  
        # Overlay mean + CI using pointplot
        sns.pointplot(
            data=group,
            order=order,
            x="smodel",
            y="score",
            markers='^',
            errorbar=("ci", 95),  # or ci='sd' for std. dev., etc.
            linestyle="none",
            color="black",
            ax=ax,
            linewidth=1,
            legend=False,
        )

        xlim = ax.get_xlim()
        ax.hlines(0.5, *xlim, linestyle="--", color="gray", linewidth=1, zorder=4)
        ax.set_xlim(*xlim)
        ax.set_title(model)
    for i, ax in enumerate(axes):
        ax.set_ylim(0, 1)
        ax.set_xlabel(None)
        # rotate the x axis tick labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        if i > 0:
            ax.set_ylabel(None)
            ax.set_yticklabels([])

    plt.tight_layout()
    if suffix is None:
        suffix = ""
    fig.savefig(root_dir / 'scripts' / f"pairwise_preference{suffix}.png")    


if __name__ == "__main__":
    analyze_pairwise_preference(dataset_filter=["cnn", "xsum"], suffix="_cnn_xsum")
    analyze_pairwise_preference(dataset_filter=["cnn"], suffix="_cnn")
    analyze_pairwise_preference(dataset_filter=["xsum"], suffix="_xsum")