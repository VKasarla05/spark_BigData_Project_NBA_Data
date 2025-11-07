# 0) Imports
import time
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql.types import DoubleType
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, auc

# ============================================================================
# SPARK SETUP
# ============================================================================

spark = SparkSession.builder.appName("NBA_GBT_Classifier").getOrCreate()

dataset_path = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
results_dir = "/home/sat3812/BD_Project/Results/GradientBoostedTrees"
os.makedirs(results_dir, exist_ok=True)

sns.set_style("whitegrid")

# ============================================================================
# DATA LOADING & TARGET PREPARATION
# ============================================================================

print("Loading dataset...")
df = spark.read.csv(dataset_path, header=True, inferSchema=True).cache()
print(f"✓ Loaded {df.count():,} rows with {len(df.columns)} features")

# Identify the performance metric column
perf_cols = [col for col in df.columns if "PER" in col.upper()]
if not perf_cols:
    raise ValueError("Dataset missing PER column")
perf_metric = perf_cols[0]
print(f"Using {perf_metric} as performance indicator")

# Create binary target: 1 if player performance is above average
avg_performance = df.select(F.mean(F.col(perf_metric))).collect()[0][0]
print(f"Average {perf_metric}: {avg_performance:.2f}")

df_with_target = df.withColumn(
    "is_above_average",
    F.when(F.col(perf_metric) >= avg_performance, 1).otherwise(0)
)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Extract all numeric columns (excluding target and metric)
numeric_features = [
    col for col, dtype in df_with_target.dtypes
    if dtype in ("int", "bigint", "float", "double")
    and col not in (perf_metric, "is_above_average")
]

print(f"Using {len(numeric_features)} numeric features")

if len(numeric_features) < 2:
    raise ValueError("Insufficient features for modeling")

# Build preprocessing pipeline
feature_builder = VectorAssembler(
    inputCols=numeric_features,
    outputCol="raw_features",
    handleInvalid="skip"
)

feature_scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

# ============================================================================
# MODEL SETUP
# ============================================================================

gbt = GBTClassifier(
    featuresCol="scaled_features",
    labelCol="is_above_average",
    maxIter=80,
    seed=42,
    subsamplingRate=0.9
)

# Hyperparameter grid for tuning
hp_grid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [4, 6, 8])
    .addGrid(gbt.maxBins, [32, 64])
    .addGrid(gbt.stepSize, [0.05, 0.1, 0.2])
    .build()
)

# Use train-validation split (faster than k-fold CV for big data)
roc_evaluator = BinaryClassificationEvaluator(
    labelCol="is_above_average",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

validator = TrainValidationSplit(
    estimator=gbt,
    estimatorParamMaps=hp_grid,
    evaluator=roc_evaluator,
    trainRatio=0.8,
    parallelism=2
)

pipeline = Pipeline(stages=[feature_builder, feature_scaler, validator])

# ============================================================================
# TRAINING
# ============================================================================

# Split data
prep_df = df_with_target.select(*numeric_features, perf_metric, "is_above_average").dropna()
train_set, test_set = prep_df.randomSplit([0.8, 0.2], seed=7)

print(f"Training samples: {train_set.count():,}")
print(f"Test samples: {test_set.count():,}")

# Train
print("Training model...")
train_start = time.time()
fitted_pipeline = pipeline.fit(train_set)
train_elapsed = time.time() - train_start
print(f"✓ Training completed in {train_elapsed:.1f}s")

# Get the best model
best_validator = fitted_pipeline.stages[-1]
best_model = best_validator.bestModel

# ============================================================================
# EVALUATION
# ============================================================================

print("Generating predictions...")
pred_start = time.time()
test_scored = fitted_pipeline.transform(test_set).cache()
pred_elapsed = time.time() - pred_start
print(f"✓ Predictions completed in {pred_elapsed:.2f}s")


# Calculate metrics
acc_eval = MulticlassClassificationEvaluator(
    labelCol="is_above_average", predictionCol="prediction", metricName="accuracy"
)
prec_eval = MulticlassClassificationEvaluator(
    labelCol="is_above_average", predictionCol="prediction", metricName="weightedPrecision"
)
rec_eval = MulticlassClassificationEvaluator(
    labelCol="is_above_average", predictionCol="prediction", metricName="weightedRecall"
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="is_above_average", predictionCol="prediction", metricName="f1"
)
roc_eval = BinaryClassificationEvaluator(
    labelCol="is_above_average", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
pr_eval = BinaryClassificationEvaluator(
    labelCol="is_above_average", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
)

results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"],
    "Score": [
        acc_eval.evaluate(test_scored),
        prec_eval.evaluate(test_scored),
        rec_eval.evaluate(test_scored),
        f1_eval.evaluate(test_scored),
        roc_eval.evaluate(test_scored),
        pr_eval.evaluate(test_scored)
    ]
})

summary = pd.DataFrame({
    "Model": ["Gradient Boosted Trees"],
    "Training_Time_sec": [train_elapsed],
    "Prediction_Time_sec": [pred_elapsed],
    "PER_Threshold": [avg_performance],
    "Test_Accuracy": [acc_eval.evaluate(test_scored)],
    "ROC_AUC": [roc_eval.evaluate(test_scored)]
})

print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)
print(results.to_string(index=False))
print("\n" + summary.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# Extract probabilities for plotting
extract_prob = F.udf(lambda vec: float(vec[1]) if vec else None, DoubleType())
test_scored_probs = test_scored.withColumn("pred_prob", extract_prob(F.col("probability")))

# Sample for visualization (avoid memory issues)
plot_data = test_scored_probs.select(
    "is_above_average", "pred_prob", "prediction"
).dropna().sample(0.5, seed=123).toPandas()

print(f"Plotting with {len(plot_data)} samples")

# ROC Curve
fpr, tpr, _ = roc_curve(plot_data["is_above_average"], plot_data["pred_prob"])
roc_score = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2.5, label=f"ROC (AUC = {roc_score:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate", fontsize=11)
plt.title("ROC Curve", fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ROC_GB.png"))
plt.close()


# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(plot_data["is_above_average"], plot_data["pred_prob"])
pr_score = auc(recall, precision)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision, linewidth=2.5, label=f"PR (AUC = {pr_score:.3f})")
plt.xlabel("Recall", fontsize=11)
plt.ylabel("Precision", fontsize=11)
plt.title("Precision-Recall Curve", fontsize=12, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Precision_Recall_GB.png"))
plt.close()


# Confusion Matrix
cm = confusion_matrix(plot_data["is_above_average"], plot_data["prediction"])
display = ConfusionMatrixDisplay(cm, display_labels=["Below Average", "Above Average"])
fig, ax = plt.subplots(figsize=(7, 6))
display.plot(ax=ax, cmap="Blues")
plt.title("Confusion Matrix", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Confusion_Matrix_GB.png"))
plt.close()


# Feature Importances
importances = np.array(best_model.featureImportances.toArray())
top_n = min(20, len(numeric_features))
top_indices = np.argsort(importances)[-top_n:][::-1]

importance_df = pd.DataFrame({
    "Feature": [numeric_features[i] for i in top_indices],
    "Importance": importances[top_indices]
})

plt.figure(figsize=(9, 7))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Top 20 Feature Importances", fontsize=12, fontweight="bold")
plt.xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Feature_GB.png"))
plt.close()

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

final_preds = test_probs.select(
    F.col(perf_metric).alias("PER"),
    F.col("is_above_average").alias("True_Label"),
    F.col("prediction").alias("Predicted_Label"),
    F.col("probability_1").alias("Confidence")
)
final_preds.write.mode("overwrite").csv(os.path.join(results_dir, "Predictions_GB"), header=True)
