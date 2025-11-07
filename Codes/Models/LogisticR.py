# =========================================================
# Spark Logistic Regression Model for NBA Player Efficiency
# =========================================================

# Import Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# =========================================================
# Spark Session
# =========================================================
spark = SparkSession.builder.appName("NBA_LogisticRegression_Classification").getOrCreate()

# =========================================================
# Load Dataset
# =========================================================
data_path = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
print(f"Loading dataset...\n✓ Loaded {df.count()} rows with {len(df.columns)} columns")

# =========================================================
# Results Directory
# =========================================================
results_dir = "/content/Results/Logistic/"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# =========================================================
# Target & Feature Engineering
# =========================================================
target_candidates = [c for c in df.columns if "PER" in c.upper()]
if not target_candidates:
    raise ValueError("❌ Target variable (PER) not found.")
target_col = target_candidates[0]
print(f"Using {target_col} as target metric")

average_per = df.select(F.mean(F.col(target_col))).collect()[0][0]
print(f"Average {target_col}: {average_per:.2f}")

df_labeled = df.withColumn(
    "label",
    F.when(F.col(target_col) >= average_per, 1).otherwise(0)
)

numeric_features = [
    c for c, t in df_labeled.dtypes
    if t in ("int", "bigint", "float", "double") and c not in (target_col, "label")
]
print(f"Using {len(numeric_features)} numeric features")

# Assemble + Scale
assembler = VectorAssembler(inputCols=numeric_features, outputCol="raw_features", handleInvalid="skip")
scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")

df_prepared = assembler.transform(df_labeled)
df_scaled = scaler.fit(df_prepared).transform(df_prepared).select("features", "label")

# Split Data
train_df, test_df = df_scaled.randomSplit([0.8, 0.2], seed=42)
print(f"Training samples: {train_df.count()} | Testing samples: {test_df.count()}")

# =========================================================
# Train Logistic Regression
# =========================================================
print("Training Logistic Regression model...")
start_time = time.time()
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=150,
    regParam=0.01,
    elasticNetParam=0.2,
    threshold=0.5
)
lr_model = lr.fit(train_df)
train_time = time.time() - start_time
print(f"✓ Training completed in {train_time:.2f}s")

# Predict
print("Generating predictions...")
pred_start = time.time()
predictions = lr_model.transform(test_df).cache()
pred_time = time.time() - pred_start
print(f"✓ Predictions completed in {pred_time:.2f}s")

# =========================================================
# Evaluate Model
# =========================================================
acc_eval  = MulticlassClassificationEvaluator(metricName="accuracy")
prec_eval = MulticlassClassificationEvaluator(metricName="weightedPrecision")
rec_eval  = MulticlassClassificationEvaluator(metricName="weightedRecall")
f1_eval   = MulticlassClassificationEvaluator(metricName="f1")
roc_eval  = BinaryClassificationEvaluator(metricName="areaUnderROC")
pr_eval   = BinaryClassificationEvaluator(metricName="areaUnderPR")

accuracy  = acc_eval.evaluate(predictions)
precision = prec_eval.evaluate(predictions)
recall    = rec_eval.evaluate(predictions)
f1_score  = f1_eval.evaluate(predictions)
roc_auc   = roc_eval.evaluate(predictions)
pr_auc    = pr_eval.evaluate(predictions)

results_df = pd.DataFrame({
    "Model": ["Logistic Regression Classifier"],
    "Accuracy":  [accuracy],
    "Precision": [precision],
    "Recall":    [recall],
    "F1-Score":  [f1_score],
    "ROC-AUC":   [roc_auc],
    "PR-AUC":    [pr_auc],
    "Train_Time(s)": [train_time],
    "Predict_Time(s)": [pred_time]
})

print("\nClassification Performance Summary:")
print(results_df.round(4))
results_df.to_csv(os.path.join(results_dir, "LogisticRegression_Results.csv"), index=False)

# =========================================================
# Confusion Matrix
# =========================================================
pred_pd = predictions.select("label", "prediction").toPandas()
cm = confusion_matrix(pred_pd["label"], pred_pd["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Below Avg", "Above Avg"])
disp.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Confusion_Matrix_LR.png"))
plt.close()

# =========================================================
# ROC & Precision–Recall Curves
# =========================================================
prob_pd = predictions.select("label", "probability").toPandas()
prob_pd["prob_1"] = prob_pd["probability"].apply(lambda v: float(v[1]))

fpr, tpr, _ = roc_curve(prob_pd["label"], prob_pd["prob_1"])
prec, rec, _ = precision_recall_curve(prob_pd["label"], prob_pd["prob_1"])
roc_auc_s = roc_auc_score(prob_pd["label"], prob_pd["prob_1"])
pr_auc_s  = average_precision_score(prob_pd["label"], prob_pd["prob_1"])

# ROC Curve
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, lw=2, color='darkgreen', label=f"AUC = {roc_auc_s:.3f}")
plt.plot([0,1], [0,1], linestyle="--", color='gray')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression", fontsize=14, weight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ROC_Curve.png"))
plt.close()

# Precision–Recall Curve
plt.figure(figsize=(7,5))
plt.plot(rec, prec, lw=2, color='darkorange', label=f"AP = {pr_auc_s:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve — Logistic Regression", fontsize=14, weight='bold')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Precision_Recall_Curve_LR.png"))
plt.close()

# =========================================================
# Save Predictions (Fixed for CSV compatibility)
# =========================================================
from pyspark.sql.types import DoubleType

# Extract probability of class 1
extract_prob = F.udf(lambda v: float(v[1]) if v is not None else None, DoubleType())
predictions_fixed = predictions.withColumn("Prob_Class1", extract_prob(F.col("probability")))

# Select final columns to save
final_preds = predictions_fixed.select(
    F.col("label").alias("True_Label"),
    F.col("prediction").alias("Predicted_Label"),
    F.col("Prob_Class1").alias("Probability_Class_1")
)

# Save clean CSV
final_preds.write.mode("overwrite").csv(os.path.join(results_dir, "Predictions_LR"), header=True)


