import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_PATH = Path("data/yellow_tripdata_2026-05_ml.parquet")
OUTPUT_DIR = Path("output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10

RC_LABELS = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau", 5: "Negotiated", 99: "Unknown"}
PALETTE = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868", 4: "#C44E52", 5: "#8172B3", 99: "#937860"}


def load_data():
    print("Loading ML dataset...")
    df = pd.read_parquet(DATA_PATH)
    df["rate_label"] = df["RatecodeID"].map(RC_LABELS)
    print(f"Shape: {df.shape}")
    return df


def plot_ratecode_distribution(df):
    rc_counts = df["RatecodeID"].value_counts().sort_index()
    labels = [RC_LABELS.get(int(k), str(k)) for k in rc_counts.index]
    colors = [PALETTE.get(int(k), "#888888") for k in rc_counts.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, rc_counts.values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, count, pct in zip(bars, rc_counts.values, rc_counts.values / len(df) * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30000,
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_title("RatecodeID Distribution", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_ylim(0, rc_counts.max() * 1.2)
    fig.tight_layout()
    path = OUTPUT_DIR / "rc_01_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plotly interactive version
    fig2 = px.bar(x=labels, y=rc_counts.values, color=labels, color_discrete_sequence=colors,
                  title="RatecodeID Distribution (Interactive)",
                  labels={"x": "RatecodeID", "y": "Count"})
    fig2.update_layout(showlegend=False, template="plotly_white")
    path2 = OUTPUT_DIR / "rc_01_distribution.html"
    fig2.write_html(path2)
    print(f"Saved: {path2}")


def plot_trip_distance_by_rc(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    order = sorted(df["RatecodeID"].unique())
    order_labels = [RC_LABELS.get(int(k), str(k)) for k in order]
    order_colors = [PALETTE.get(int(k), "#888888") for k in order]

    # Clip extreme outliers for visibility
    df_plot = df.copy()
    q99 = df_plot["trip_distance"].quantile(0.99)
    df_plot["trip_distance_clipped"] = df_plot["trip_distance"].clip(upper=q99)

    bp = ax.boxplot(
        [df_plot[df_plot["RatecodeID"] == rc]["trip_distance_clipped"] for rc in order],
        patch_artist=True, widths=0.6,
        showfliers=True, flierprops=dict(marker="o", markersize=1.5, alpha=0.3, markerfacecolor="red"),
        medianprops=dict(color="white", linewidth=1.5))
    ax.set_xticklabels(order_labels)
    for patch, color in zip(bp["boxes"], order_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("trip_distance by RatecodeID (clipped at 99th percentile)", fontsize=14)
    ax.set_ylabel("Trip Distance (miles)")
    ax.set_xlabel("RatecodeID")
    fig.tight_layout()
    path = OUTPUT_DIR / "rc_02_trip_distance_box.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Violin plot with KDE
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    parts = ax2.violinplot(
        [df_plot[df_plot["RatecodeID"] == rc]["trip_distance_clipped"] for rc in order],
        positions=range(1, len(order) + 1), showmeans=True, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(order_colors[i])
        pc.set_alpha(0.5)
    ax2.set_xticks(range(1, len(order) + 1))
    ax2.set_xticklabels(order_labels)
    ax2.set_title("trip_distance Distribution by RatecodeID (Violin + Median)", fontsize=14)
    ax2.set_ylabel("Trip Distance (miles)")
    fig2.tight_layout()
    path2 = OUTPUT_DIR / "rc_03_trip_distance_violin.png"
    fig2.savefig(path2, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {path2}")

    # Plotly interactive box
    df_plot["rate_label"] = df_plot["RatecodeID"].map(RC_LABELS)
    fig3 = px.box(df_plot, x="rate_label", y="trip_distance_clipped", color="rate_label",
                  color_discrete_map={RC_LABELS.get(int(k), str(k)): PALETTE.get(int(k), "#888")
                                      for k in order},
                  title="trip_distance by RatecodeID (Interactive, clipped 99th)",
                  labels={"trip_distance_clipped": "Trip Distance (miles)", "rate_label": "RatecodeID"},
                  template="plotly_white")
    fig3.update_layout(showlegend=False, height=500)
    path3 = OUTPUT_DIR / "rc_03_trip_distance_box.html"
    fig3.write_html(path3)
    print(f"Saved: {path3}")


def plot_passenger_count_by_rc(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = sorted(df["RatecodeID"].unique())
    order_labels = [RC_LABELS.get(int(k), str(k)) for k in order]
    order_colors = [PALETTE.get(int(k), "#888888") for k in order]

    # Bar: mean passenger count
    means = [df[df["RatecodeID"] == rc]["passenger_count"].mean() for rc in order]
    axes[0].bar(order_labels, means, color=order_colors, edgecolor="white")
    for i, (bar, m) in enumerate(zip(axes[0].patches, means)):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{m:.2f}", ha="center", fontsize=9)
    axes[0].set_title("Mean passenger_count by RatecodeID")
    axes[0].set_ylabel("Average Passengers")
    axes[0].set_ylim(0, max(means) * 1.15)

    # Stacked bar: passenger count distribution
    for rc in order:
        s = df[df["RatecodeID"] == rc]
        pcts = [s[s["passenger_count"] == p].shape[0] / s.shape[0] * 100 for p in range(1, 9)]
        for p_idx, pct in enumerate(pcts):
            if pct > 0:
                axes[1].bar(order_labels[order.index(rc)], pct,
                            bottom=sum(pcts[:p_idx]) if p_idx > 0 else 0,
                            color=sns.color_palette("Blues", 8)[p_idx],
                            edgecolor="white", linewidth=0.3,
                            label=f"{p_idx+1}" if rc == order[0] else "")
    axes[1].set_title("passenger_count Distribution by RatecodeID")
    axes[1].set_ylabel("Proportion (%)")
    axes[1].legend(title="Passengers", fontsize=7, title_fontsize=8)
    fig.tight_layout()
    path = OUTPUT_DIR / "rc_04_passenger_count.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plotly: passenger_count histogram by RC
    fig2 = px.histogram(df, x="passenger_count", color="rate_label",
                        color_discrete_map={RC_LABELS.get(int(k), str(k)): PALETTE.get(int(k), "#888")
                                            for k in order},
                        barmode="group", nbins=8,
                        title="passenger_count Histogram by RatecodeID",
                        labels={"passenger_count": "Passenger Count", "rate_label": "RatecodeID"},
                        template="plotly_white")
    fig2.update_layout(height=450, legend_title="RatecodeID")
    path2 = OUTPUT_DIR / "rc_04_passenger_count.html"
    fig2.write_html(path2)
    print(f"Saved: {path2}")


def plot_hour_by_rc(df):
    order = sorted(df["RatecodeID"].unique())
    order_labels = [RC_LABELS.get(int(k), str(k)) for k in order]
    order_colors = [PALETTE.get(int(k), "#888888") for k in order]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, rc in enumerate(order):
        s = df[df["RatecodeID"] == rc]
        ax = axes[i]
        hour_counts = s["hour"].value_counts().sort_index()
        ax.bar(hour_counts.index, hour_counts.values, color=order_colors[i], alpha=0.8, edgecolor="white")
        ax.axvline(s["hour"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={s['hour'].mean():.1f}")
        ax.set_title(f"{order_labels[i]} (n={len(s):,})")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Count")
        ax.set_xticks(range(0, 24, 4))
        ax.legend(fontsize=7)
    axes[5].set_visible(False)
    fig.suptitle("Pickup Hour Distribution by RatecodeID", fontsize=14, y=1.01)
    fig.tight_layout()
    path = OUTPUT_DIR / "rc_05_hour_by_rc.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plotly: interactive heatmap (hour x RatecodeID)
    heatmap_data = df.pivot_table(index="RatecodeID", columns="hour", values="trip_distance", aggfunc="count")
    heatmap_data.index = [RC_LABELS.get(int(k), str(k)) for k in heatmap_data.index]
    fig2 = px.imshow(heatmap_data, aspect="auto", color_continuous_scale="Blues",
                     title="Trip Count Heatmap: Hour × RatecodeID",
                     labels={"x": "Hour", "y": "RatecodeID", "color": "Trip Count"},
                     template="plotly_white")
    fig2.update_layout(height=400)
    path2 = OUTPUT_DIR / "rc_05_hour_heatmap.html"
    fig2.write_html(path2)
    print(f"Saved: {path2}")


def plot_correlation_heatmap(df):
    numeric_cols = ["trip_distance", "passenger_count", "hour"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Numeric Feature Correlation Heatmap")
    fig.tight_layout()
    path = OUTPUT_DIR / "rc_06_correlation_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_categorical_by_rc(df):
    cat_cols = ["VendorID", "payment_type", "is_weekend", "store_and_fwd_flag"]
    order = sorted(df["RatecodeID"].unique())

    # Stacked proportion bar for each categorical feature
    for col in cat_cols:
        data = {}
        for rc in order:
            s = df[df["RatecodeID"] == rc]
            vc = s[col].value_counts(normalize=True)
            data[RC_LABELS.get(int(rc), str(rc))] = vc
        prop_df = pd.DataFrame(data).T.fillna(0) * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        prop_df.plot(kind="bar", stacked=True, ax=ax, colormap="Set2", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{col} by RatecodeID (% stacked)", fontsize=13)
        ax.set_ylabel("Proportion (%)")
        ax.set_xlabel("RatecodeID")
        ax.legend(title=col, fontsize=8, title_fontsize=9)
        ax.set_ylim(0, 105)
        fig.tight_layout()
        path = OUTPUT_DIR / f"rc_07_{col}_by_rc.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Plotly interactive: group bar for categorical
    sample = df.sample(n=min(20000, len(df)), random_state=42)
    sample["rate_label"] = sample["RatecodeID"].map(RC_LABELS)
    fig2 = px.histogram(sample, x="payment_type", color="rate_label", barmode="group",
                        title="payment_type by RatecodeID (Interactive, 20k sample)",
                        labels={"rate_label": "RatecodeID"}, template="plotly_white")
    fig2.update_layout(height=450)
    path2 = OUTPUT_DIR / "rc_07_payment_type.html"
    fig2.write_html(path2)
    print(f"Saved: {path2}")


def plot_eta_squared_chart(df):
    """ANOVA eta-squared bar chart for feature importance"""
    numeric_cols = ["trip_distance", "passenger_count", "hour"]
    grand_mean = df[numeric_cols].mean()
    ss_between = {}
    ss_total = {}

    for col in numeric_cols:
        ss_b = 0
        ss_t = 0
        for rc, group in df.groupby("RatecodeID"):
            n_k = len(group)
            mean_k = group[col].mean()
            ss_b += n_k * (mean_k - grand_mean[col]) ** 2
            ss_t += ((group[col] - grand_mean[col]) ** 2).sum()
        ss_between[col] = ss_b
        ss_total[col] = ss_t

    eta2 = {col: ss_between[col] / ss_total[col] for col in numeric_cols}

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(eta2.keys(), eta2.values(), color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="white")
    for bar, val in zip(bars, eta2.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=10)
    ax.set_title("ANOVA η² (Effect Size): Numeric Features ~ RatecodeID")
    ax.set_ylabel("η² (Eta-squared)")
    ax.set_ylim(0, max(eta2.values()) * 1.15)
    fig.tight_layout()
    path = OUTPUT_DIR / "rc_08_eta_squared.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_t_test_summary():
    """t-test results summary chart"""
    comparisons = [
        ("Standard vs JFK", "trip_distance", -1251.4),
        ("Standard vs Unknown", "trip_distance", -373.5),
        ("JFK vs Newark", "trip_distance", 60.4),
        ("Standard vs Negotiated", "trip_distance", -82.1),
        ("Standard vs Unknown", "passenger_count", 649.0),
        ("Standard vs JFK", "passenger_count", -85.4),
        ("Standard vs Unknown", "hour", 214.0),
        ("Standard vs Negotiated", "hour", 41.0),
        ("JFK vs Newark", "hour", 22.9),
    ]

    labels = [f"{c[0]}\n({c[1]})" for c in comparisons]
    values = [c[2] for c in comparisons]
    colors = ["#4C72B0" if v > 0 else "#C44E52" for v in values]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels, [abs(v) for v in values], color=colors, edgecolor="white")
    for i, (v, c) in enumerate(zip(values, colors)):
        ax.text(abs(v) + 15, i, f"t={v:+.1f}", va="center", fontsize=8, color=c)
    ax.set_title("Welch's t-test: |t-statistic| by Group Comparison (all p<0.001)")
    ax.set_xlabel("|t-statistic|")
    ax.invert_yaxis()
    fig.tight_layout()
    path = OUTPUT_DIR / "rc_09_ttest_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    df = load_data()

    print("\n--- 1. RatecodeID Distribution ---")
    plot_ratecode_distribution(df)

    print("\n--- 2. trip_distance Analysis ---")
    plot_trip_distance_by_rc(df)

    print("\n--- 3. passenger_count Analysis ---")
    plot_passenger_count_by_rc(df)

    print("\n--- 4. Hour Analysis ---")
    plot_hour_by_rc(df)

    print("\n--- 5. Correlation Heatmap ---")
    plot_correlation_heatmap(df)

    print("\n--- 6. Categorical Feature Analysis ---")
    plot_categorical_by_rc(df)

    print("\n--- 7. Eta-squared Chart ---")
    plot_eta_squared_chart(df)

    print("\n--- 8. t-test Summary Chart ---")
    plot_t_test_summary()

    print(f"\nAll visualizations saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
