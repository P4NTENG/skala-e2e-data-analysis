import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

DATA_PATH = Path("data/yellow_tripdata_2026-05.parquet")
OUTPUT_DIR = Path("output/figures")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def load_data():
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = pd.read_parquet(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    return df


def print_summary_stats(df):
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS (NUMERIC COLUMNS)")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(desc.to_string())

    print("\n" + "=" * 60)
    print("SKEWNESS & KURTOSIS")
    print("=" * 60)
    for col in numeric_cols:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        print(f"  {col:30s} skew={skew:+.4f}  kurtosis={kurt:+.4f}")


def plot_numeric_seaborn(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(numeric_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row

    # Histograms + KDE
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 3.5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        data = df[col].dropna()
        ax = axes[i]
        # Clip to 1st-99th percentile for better view
        lo, hi = np.percentile(data, [1, 99])
        clipped = data.clip(lo, hi)
        ax.hist(clipped, bins=80, density=True, alpha=0.6, color="steelblue", edgecolor="white")
        sns.kdeplot(clipped, ax=ax, color="darkorange", linewidth=2)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Column Distributions (Histogram + KDE, clipped 1-99%)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = OUTPUT_DIR / "01_numeric_hist_kde.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_box_seaborn(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n = len(numeric_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 3.5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.boxplot(df[col].dropna(), orientation="vertical", patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6),
                   medianprops=dict(color="darkorange", linewidth=2),
                   flierprops=dict(marker="o", markerfacecolor="red", markersize=2, alpha=0.3))
        ax.set_title(col, fontsize=10)
        ax.set_xticklabels([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Column Box Plots (Outlier Detection)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = OUTPUT_DIR / "02_numeric_boxplot.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corp = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corp, dtype=bool))
    sns.heatmap(corp, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Numeric Column Correlation Heatmap", fontsize=14)
    fig.tight_layout()
    path = OUTPUT_DIR / "03_correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_categorical_distribution(df):
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if not cat_cols:
        print("No categorical columns found.")
        return

    for col in cat_cols:
        value_counts = df[col].value_counts().nlargest(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(value_counts)), value_counts.values, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{col} — Top {len(value_counts)} categories", fontsize=12)
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = OUTPUT_DIR / f"04_cat_{col}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plotly_interactive_histogram(df, col):
    data = df[col].dropna()
    lo, hi = np.percentile(data, [0.1, 99.9])
    clipped = data.clip(lo, hi)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clipped, nbinsx=100, name=col,
                               marker_color="steelblue", opacity=0.75))
    fig.update_layout(
        title=f"Interactive Distribution: {col} (clipped 0.1-99.9%)",
        xaxis_title=col, yaxis_title="Count",
        template="plotly_white", height=500
    )
    path = OUTPUT_DIR / f"05_plotly_{col}.html"
    fig.write_html(path)
    print(f"Saved: {path}")


def plotly_box_by_category(df, numeric_col, cat_col):
    data = df[[numeric_col, cat_col]].dropna()
    lo, hi = data[numeric_col].quantile([0.01, 0.99])
    data = data[(data[numeric_col] >= lo) & (data[numeric_col] <= hi)]

    fig = px.box(data, x=cat_col, y=numeric_col,
                 title=f"{numeric_col} by {cat_col} (outliers trimmed 1-99%)",
                 template="plotly_white", color=cat_col,
                 category_orders={cat_col: sorted(data[cat_col].unique())})
    fig.update_layout(height=500, showlegend=False,
                      xaxis_tickangle=-45)
    path = OUTPUT_DIR / f"06_plotly_box_{numeric_col}_by_{cat_col}.html"
    fig.write_html(path)
    print(f"Saved: {path}")


def detect_outliers_iqr(df):
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION (IQR Method)")
    print("=" * 60)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}
    for col in numeric_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        lower_count = (data < lower).sum()
        upper_count = (data > upper).sum()
        total_outliers = lower_count + upper_count
        pct = total_outliers / len(data) * 100
        outlier_summary[col] = {
            "Q1": Q1, "Q3": Q3, "IQR": IQR,
            "lower_fence": lower, "upper_fence": upper,
            "lower_outliers": lower_count, "upper_outliers": upper_count,
            "total_outliers": total_outliers, "pct_outliers": pct
        }
        print(f"  {col:30s}  lower={lower_count:>8,}  upper={upper_count:>8,}  "
              f"total={total_outliers:>8,}  ({pct:.2f}%)")
    return outlier_summary


def main():
    df = load_data()
    print_summary_stats(df)

    print("\n" + "=" * 60)
    print("GENERATING STATIC CHARTS (Seaborn/Matplotlib)")
    print("=" * 60)
    plot_numeric_seaborn(df)
    plot_box_seaborn(df)
    plot_correlation_heatmap(df)
    plot_categorical_distribution(df)

    print("\n" + "=" * 60)
    print("GENERATING INTERACTIVE CHARTS (Plotly)")
    print("=" * 60)
    # Pick key numeric columns for interactive viz
    key_numeric = ["trip_distance", "fare_amount", "total_amount", "tip_amount",
                   "tolls_amount", "passenger_count"]
    for col in key_numeric:
        if col in df.columns:
            plotly_interactive_histogram(df, col)

    # Box plot by payment_type (if exists)
    if "payment_type" in df.columns and "total_amount" in df.columns:
        plotly_box_by_category(df, "total_amount", "payment_type")

    detect_outliers_iqr(df)

    print("\n" + "=" * 60)
    print("ALL DONE. Check output/figures/ for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
