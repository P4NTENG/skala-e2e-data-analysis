import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, balanced_accuracy_score,
                              precision_recall_fscore_support)
import joblib
from pathlib import Path
import time

DATA_PATH = Path("data/yellow_tripdata_2026-05_ml.parquet")
OUTPUT_DIR = Path("output")
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

RC_LABELS = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau", 5: "Negotiated", 99: "Unknown"}

NUMERIC_FEATURES = ["passenger_count", "hour", "day_of_week", "is_weekend"]
LOW_CARDINALITY = ["VendorID", "payment_type", "store_and_fwd_flag"]
HIGH_CARDINALITY = ["PULocationID", "DOLocationID"]
TARGET = "RatecodeID"


def load_and_split(sample_size=None):
    print("Loading ML dataset...")
    df = pd.read_parquet(DATA_PATH)

    for col in LOW_CARDINALITY + HIGH_CARDINALITY:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")

    if sample_size and sample_size < len(df):
        # Use fixed seed for reproducibility
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"Sampled {sample_size:,} rows from {len(df):,}")

    X = df[NUMERIC_FEATURES + LOW_CARDINALITY + HIGH_CARDINALITY]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Features: {X.shape[1]}  Classes: {y.nunique()}")
    return X_train, X_test, y_train, y_test


def build_pipeline_rf():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat_low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), LOW_CARDINALITY),
            ("cat_high", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), HIGH_CARDINALITY),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=50,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )
    return pipeline


def build_pipeline_hgb():
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_low", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), LOW_CARDINALITY),
            ("cat_high", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), HIGH_CARDINALITY),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=15,
                learning_rate=0.1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]
    )
    return pipeline


def evaluate_model(pipeline, X_train, y_train, X_test, y_test, model_name):
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 60}")

    # Cross-validation on training set
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    start = time.time()
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1
    )
    cv_time = time.time() - start
    print(f"CV Balanced Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"CV Time: {cv_time:.1f}s")

    # Train on full training set
    print("\n--- Training on full train set ---")
    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Train Time: {train_time:.1f}s")

    # Predict
    start = time.time()
    y_pred = pipeline.predict(X_test)
    pred_time = time.time() - start
    print(f"Predict Time ({len(X_test):,} samples): {pred_time:.1f}s")

    # Evaluation
    print(f"\n--- Test Set Evaluation ---")
    class_names = [RC_LABELS.get(c, str(c)) for c in sorted(y_test.unique())]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")

    # Feature importance (if available)
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_low_feature_names = list(
            preprocessor.named_transformers_["cat_low"].get_feature_names_out(LOW_CARDINALITY)
        )
        all_feature_names = NUMERIC_FEATURES + cat_low_feature_names + HIGH_CARDINALITY
        importances = classifier.feature_importances_

        if len(all_feature_names) == len(importances):
            print("\n--- Top 10 Feature Importances ---")
            sorted_idx = np.argsort(importances)[::-1]
            for i in sorted_idx[:10]:
                print(f"  {importances[i]:.4f}  {all_feature_names[i]}")

    return y_test, y_pred


def plot_confusion_matrix(y_true, y_pred, model_name, class_names):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (recall)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        [f"Confusion Matrix — {model_name}", f"Normalized Confusion Matrix (Recall) — {model_name}"],
        ["d", ".2f"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={"shrink": 0.8}, vmin=0, vmax=data.max() if fmt == "d" else 1.0)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.tight_layout()
    alias = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / f"ml_{alias}_confusion.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plotly interactive
    fig2 = px.imshow(cm_norm, text_auto=".2f", aspect="auto", color_continuous_scale="Blues",
                     x=class_names, y=class_names, zmin=0, zmax=1,
                     title=f"Normalized Confusion Matrix — {model_name} (Interactive)")
    fig2.update_layout(height=500)
    path2 = FIGURES_DIR / f"ml_{alias}_confusion.html"
    fig2.write_html(path2)
    print(f"Saved: {path2}")


def plot_class_metrics_comparison(y_test, y_pred_rf, y_pred_hgb, class_names):
    _, _, f1_rf, _ = precision_recall_fscore_support(y_test, y_pred_rf, labels=sorted(y_test.unique()))
    _, _, f1_hgb, _ = precision_recall_fscore_support(y_test, y_pred_hgb, labels=sorted(y_test.unique()))

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, f1_rf, width, label="RandomForest", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width / 2, f1_hgb, width, label="HistGradientBoosting", color="#DD8452", edgecolor="white")

    for bar, val in zip(bars1, f1_rf):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}",
                ha="center", fontsize=8, color="#4C72B0")
    for bar, val in zip(bars2, f1_hgb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.2f}",
                ha="center", fontsize=8, color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1-Score Comparison: RF vs HGB")
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    path = FIGURES_DIR / "ml_f1_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plotly groups bar
    df_plot = pd.DataFrame({
        "Class": class_names * 2,
        "F1": list(f1_rf) + list(f1_hgb),
        "Model": ["RandomForest"] * len(class_names) + ["HistGradientBoosting"] * len(class_names),
    })
    fig2 = px.bar(df_plot, x="Class", y="F1", color="Model", barmode="group",
                  title="Per-Class F1-Score Comparison (Interactive)",
                  template="plotly_white",
                  color_discrete_map={"RandomForest": "#4C72B0", "HistGradientBoosting": "#DD8452"})
    fig2.update_layout(height=450, yaxis_range=[0, 1.05])
    path2 = FIGURES_DIR / "ml_f1_comparison.html"
    fig2.write_html(path2)
    print(f"Saved: {path2}")


def plot_precision_recall_radar(y_test, y_pred, class_names, model_name):
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=sorted(y_test.unique()))

    categories = list(class_names) + [class_names[0]]
    prec_vals = list(prec) + [prec[0]]
    rec_vals = list(rec) + [rec[0]]
    f1_vals = list(f1) + [f1[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=prec_vals, theta=categories, fill="toself", name="Precision",
                                  line=dict(color="#4C72B0")))
    fig.add_trace(go.Scatterpolar(r=rec_vals, theta=categories, fill="toself", name="Recall",
                                  line=dict(color="#DD8452")))
    fig.add_trace(go.Scatterpolar(r=f1_vals, theta=categories, fill="toself", name="F1-Score",
                                  line=dict(color="#55A868")))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Precision / Recall / F1 Radar — {model_name}",
        template="plotly_white",
    )
    alias = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / f"ml_{alias}_radar.html"
    fig.write_html(path)
    print(f"Saved: {path}")


def plot_feature_importance(pipeline, model_name):
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        print(f"  [SKIP] {model_name}: no feature_importances_")
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    importances = classifier.feature_importances_

    # Use ColumnTransformer.get_feature_names_out() for correct order
    try:
        all_features = preprocessor.get_feature_names_out()
        feature_names = [f.replace("num__", "").replace("cat_low__", "").replace("cat_high__", "") for f in all_features]
    except Exception:
        try:
            cat_low_names = list(preprocessor.named_transformers_["cat_low"].get_feature_names_out(LOW_CARDINALITY))
        except Exception:
            cat_low_names = LOW_CARDINALITY
        feature_names = NUMERIC_FEATURES + list(cat_low_names) + HIGH_CARDINALITY

    if len(feature_names) != len(importances):
        print(f"  [SKIP] {model_name}: feature name mismatch ({len(feature_names)} vs {len(importances)})")
        return

    sorted_idx = np.argsort(importances)
    top_n = min(15, len(sorted_idx))
    top_idx = sorted_idx[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Blues(0.3 + 0.7 * importances[top_idx] / importances[top_idx].max())
    ax.barh(range(top_n), importances[top_idx], color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.invert_yaxis()
    fig.tight_layout()
    alias = model_name.lower().replace(" ", "_")
    path = FIGURES_DIR / f"ml_{alias}_feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plotly horizontal bar
    fig2 = px.bar(
        x=importances[top_idx], y=[feature_names[i] for i in top_idx],
        orientation="h", title=f"Feature Importance — {model_name} (Interactive)",
        labels={"x": "Importance", "y": "Feature"}, template="plotly_white",
    )
    fig2.update_traces(marker_color="#4C72B0")
    fig2.update_layout(height=450)
    path2 = FIGURES_DIR / f"ml_{alias}_feature_importance.html"
    fig2.write_html(path2)
    print(f"Saved: {path2}")


def plot_cv_comparison(cv_rf, cv_hgb):
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = [1, 2]
    parts = ax.boxplot([cv_rf, cv_hgb], positions=positions, widths=0.4, patch_artist=True,
                        medianprops=dict(color="white", linewidth=1.5))
    parts["boxes"][0].set_facecolor("#4C72B0")
    parts["boxes"][1].set_facecolor("#DD8452")

    # Add individual points
    for pos, scores, color in zip(positions, [cv_rf, cv_hgb], ["#4C72B0", "#DD8452"]):
        jitter = np.random.normal(0, 0.04, size=len(scores))
        ax.scatter([pos] * len(scores) + jitter, scores, alpha=0.6, color=color, s=30, edgecolors="white")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["RandomForest", "HistGradientBoosting"])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("5-Fold CV Balanced Accuracy Comparison")
    ax.set_ylim(0.7, 0.9)
    fig.tight_layout()
    path = FIGURES_DIR / "ml_cv_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_model_summary_table(bal_acc_rf, bal_acc_hgb, cv_rf, cv_hgb):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")

    data = [
        ["Balanced Accuracy (CV mean)", f"{cv_rf.mean():.4f}", f"{cv_hgb.mean():.4f}"],
        ["Balanced Accuracy (Test)", f"{bal_acc_rf:.4f}", f"{bal_acc_hgb:.4f}"],
        ["CV Std Dev", f"{cv_rf.std():.4f}", f"{cv_hgb.std():.4f}"],
    ]
    columns = ["Metric", "RandomForest", "HistGradientBoosting"]
    table = ax.table(cellText=data, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if key[0] == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif key[1] == 1:
            cell.set_facecolor("#EBF0F7")
            cell.set_text_props(fontweight="bold")

    ax.set_title("Model Performance Summary", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    path = FIGURES_DIR / "ml_summary_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    X_train, X_test, y_train, y_test = load_and_split(sample_size=500_000)

    # ---- Model 1: RandomForest ----
    pipeline_rf = build_pipeline_rf()
    y_test_rf, y_pred_rf = evaluate_model(pipeline_rf, X_train, y_train, X_test, y_test, "RandomForest")

    # ---- Model 2: HistGradientBoosting ----
    pipeline_hgb = build_pipeline_hgb()
    y_test_hgb, y_pred_hgb = evaluate_model(pipeline_hgb, X_train, y_train, X_test, y_test, "HistGradientBoosting")

    # ---- Save best model ----
    print(f"\n{'=' * 60}")
    print("Saving Models")
    print(f"{'=' * 60}")
    joblib.dump(pipeline_rf, OUTPUT_DIR / "model_rf.joblib")
    print(f"Saved: output/model_rf.joblib")
    joblib.dump(pipeline_hgb, OUTPUT_DIR / "model_hgb.joblib")
    print(f"Saved: output/model_hgb.joblib")

    # ---- Visualization ----
    print(f"\n{'=' * 60}")
    print("Generating Model Visualizations")
    print(f"{'=' * 60}")

    class_names = [RC_LABELS.get(c, str(c)) for c in sorted(y_test.unique())]

    # 1. Confusion Matrix (both models)
    plot_confusion_matrix(y_test_rf, y_pred_rf, "RandomForest", class_names)
    plot_confusion_matrix(y_test_hgb, y_pred_hgb, "HistGradientBoosting", class_names)

    # 2. Per-class F1 comparison
    plot_class_metrics_comparison(y_test, y_pred_rf, y_pred_hgb, class_names)

    # 3. Radar chart (HGB)
    plot_precision_recall_radar(y_test_hgb, y_pred_hgb, class_names, "HistGradientBoosting")

    # 4. Feature importance (RF only; HGB does not expose feature_importances_ in this sklearn version)
    plot_feature_importance(pipeline_rf, "RandomForest")
    print("  [SKIP] HistGradientBoosting: feature_importances_ not available, see RandomForest plot")

    # 5. CV comparison
    # Rerun quick CV to get per-fold scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rf = cross_val_score(pipeline_rf, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    cv_hgb = cross_val_score(pipeline_hgb, X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1)

    plot_cv_comparison(cv_rf, cv_hgb)
    plot_model_summary_table(
        balanced_accuracy_score(y_test_rf, y_pred_rf),
        balanced_accuracy_score(y_test_hgb, y_pred_hgb),
        cv_rf, cv_hgb,
    )

    print(f"\nAll visualizations saved to {FIGURES_DIR.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
