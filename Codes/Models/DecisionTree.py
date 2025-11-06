# =========================================================
# Spark Decision Tree Classification Model for NBA Player Efficiency
# =========================================================

# Import Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, classification_report
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ---------------------------------------------------------
# Start Spark Session
# ---------------------------------------------------------
spark = SparkSession.builder.appName("NBA_DT_Classification").getOrCreate()

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
# Change this to your actual path if different
data_path = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
nba_data = spark.read.csv(data_path, header=True, inferSchema=True)
print(f" Dataset loaded: {nba_data.count()} rows, {len(nba_data.columns)} columns")

# Show columns (useful for debugging column name issues)
# print(nba_data.columns)

# =========================================================
# Feature Preparation + Robust Target/Label Selection
# =========================================================

# Prefer a ready-made categorical target if available; otherwise use numeric PER and binarize
candidate_targets = ["ss_PER_quantile", "ss_PER", "PER"]
target = next((c for c in candidate_targets if c in nba_data.columns), None)
if target is None:
    raise ValueError("No PER-like target found. Expected one of: 'ss_PER_quantile', 'ss_PER', or 'PER'.")

print(f"Target selected: {target}")

# Numeric feature columns (exclude the chosen target)
feature_columns = [
    col for col, dtype in nba_data.dtypes
    if dtype in ("int", "double", "float", "bigint") and col != target
]
print(f"#Features: {len(feature_columns)}")

# Vectorize features on the same DF that will carry the label
assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
data_with_features = assembler.transform(nba_data.na.drop(subset=[target]))

# Create label:
# - If ss_PER_quantile exists => use it as integer multiclass label
# - Else (ss_PER/PER) => binary: >= average => 1, else 0
if target == "ss_PER_quantile":
    labeled = data_with_features.withColumn("label", F.col(target).cast("int"))
    label_note = "Using existing quantile bins as multiclass labels."
    average_per = None
else:
    average_per = nba_data.select(F.mean(F.col(target))).collect()[0][0]
    labeled = data_with_features.withColumn("label", F.when(F.col(target) >= average_per, 1).otherwise(0))
    label_note = f"Binarized '{target}' at its average ({average_per:.4f})."

print(f"Label prep: {label_note}")

# Standardize features — not required for trees, but kept for consistent pipeline style
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
final_df = scaler.fit(labeled).transform(labeled).select("features", "label")

# Split dataset into training and testing
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_data.count()}  |  Test: {test_data.count()}")

# =========================================================
# Visualize PER Distribution with Threshold (if using numeric PER)
# =========================================================
if average_per is not None:
    per_distribution = nba_data.select(target).toPandas()
    plt.figure(figsize=(9, 5))
    sns.histplot(per_distribution[target], bins=35, kde=True, edgecolor='black')
    plt.axvline(average_per, linestyle='--', linewidth=2, label=f'Threshold = {average_per:.2f}')
    plt.title(f"{target} Distribution", fontsize=14, weight='bold')
    plt.xlabel(target, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================================================
# Train Decision Tree Classifier
# =========================================================
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxDepth=12,           # tune as needed
    minInstancesPerNode=1,
    minInfoGain=0.0,
    impurity="gini",       # or "entropy"
    seed=42
)

start_time = time.time()
dt_model = dt.fit(train_data)
predictions = dt_model.transform(test_data).cache()
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# =========================================================
# Evaluate Model Performance (Accuracy / Precision / Recall / F1 / AUC)
# =========================================================
e_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
e_f1  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
e_wp  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
e_wr  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

results = {
    "Model": ["Decision Tree Classifier"],
    "Accuracy":  [e_acc.evaluate(predictions)],
    "Precision": [e_wp.evaluate(predictions)],
    "Recall":    [e_wr.evaluate(predictions)],
    "F1":        [e_f1.evaluate(predictions)]
}

# Binary AUC via Spark evaluator (falls back gracefully for multiclass)
try:
    auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    results["AUC"] = [auc_eval.evaluate(predictions)]
except Exception:
    results["AUC"] = [np.nan]

results_df = pd.DataFrame(results)
print("\nClassification Performance:")
try:
    display(results_df.style.background_gradient(cmap="Blues").format("{:.3f}", subset=["Accuracy","Precision","Recall","F1","AUC"]))
except NameError:
    print(results_df.round(3))

# =========================================================
# Confusion Matrix Visualization
# =========================================================
preds_pd = predictions.select("label", "prediction").toPandas()
cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Decision Tree Classification — Confusion Matrix", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# =========================================================
# ROC & Precision–Recall Curves
# =========================================================
# Convert probability column to pandas-friendly probabilities
prob_pd = predictions.select("label", "probability").toPandas()
# Handle binary vs multiclass plotting
try:
    # Attempt to pull class-1 probability (binary)
    pos_scores = prob_pd["probability"].apply(lambda v: float(v[1]))
    roc_auc = roc_auc_score(prob_pd["label"], pos_scores)
    pr_auc  = average_precision_score(prob_pd["label"], pos_scores)

    fpr, tpr, _ = roc_curve(prob_pd["label"], pos_scores)
    prec, rec, _ = precision_recall_curve(prob_pd["label"], pos_scores)

    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Decision Tree")
    plt.legend(loc="lower right")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(rec, prec, lw=2, label=f"AP = {pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — Decision Tree")
    plt.legend(loc="lower left")
    plt.tight_layout(); plt.show()

except Exception:
    # Multiclass: compute macro-averaged AUCs using OvR one-vs-rest
    # Extract full probability vectors as a matrix
    probs_mat = np.vstack(prob_pd["probability"].apply(lambda v: np.array(v.toArray() if hasattr(v, "toArray") else v)))
    y_true = preds_pd["label"].values.astype(int)
    num_classes = probs_mat.shape[1]

    # Macro ROC-AUC
    try:
        roc_auc_macro = roc_auc_score(y_true, probs_mat, multi_class="ovr", average="macro")
    except Exception:
        roc_auc_macro = np.nan

    # Macro PR-AUC (average per-class AP)
    pr_macro = 0.0
    for c in range(num_classes):
        y_bin = (y_true == c).astype(int)
        try:
            pr_macro += average_precision_score(y_bin, probs_mat[:, c])
        except Exception:
            pass
    pr_macro = pr_macro / num_classes if num_classes > 0 else np.nan

    print(f"(Multiclass) ROC AUC (OvR, macro): {roc_auc_macro if roc_auc_macro==roc_auc_macro else 'N/A'}")
    print(f"(Multiclass) PR  AUC (macro avg): {pr_macro if pr_macro==pr_macro else 'N/A'}")

    # Micro-average ROC/PR curves (single curve)
    try:
        y_true_onehot = np.eye(num_classes)[y_true]
        fpr_m, tpr_m, _ = roc_curve(y_true_onehot.ravel(), probs_mat.ravel())

        plt.figure(figsize=(7,5))
        plt.plot(fpr_m, tpr_m, lw=2); plt.plot([0,1], [0,1], linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (micro-average) — Decision Tree")
        plt.tight_layout(); plt.show()

        prec_m, rec_m, _ = precision_recall_curve(y_true_onehot.ravel(), probs_mat.ravel())
        plt.figure(figsize=(7,5))
        plt.plot(rec_m, prec_m, lw=2)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (micro-average) — Decision Tree")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"(Multiclass ROC/PR curves skipped): {e}")

# =========================================================
# Detailed Metrics (macro/micro/weighted) + Report
# =========================================================
y_true = preds_pd["label"].values.astype(int)
y_pred = preds_pd["prediction"].values.astype(int)

prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro',  zero_division=0)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro',  zero_division=0)
prec_w,    rec_w,    f1_w,    _   = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

print("\n=== Detailed Metrics ===")
print(f"Precision (macro): {prec_macro:.4f} | Recall (macro): {rec_macro:.4f} | F1 (macro): {f1_macro:.4f}")
print(f"Precision (micro): {prec_micro:.4f} | Recall (micro): {rec_micro:.4f} | F1 (micro): {f1_micro:.4f}")
print(f"Precision (wgt)  : {prec_w:.4f} | Recall (wgt)  : {rec_w:.4f} | F1 (wgt)  : {f1_w:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4, zero_division=0))

