# =========================================================
# Spark Random Forest Classification Model for NBA Player Efficiency
# =========================================================

# Import Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime

# =========================================================
# Spark Session
# =========================================================
spark = SparkSession.builder.appName("NBA_RF_Classification").getOrCreate()

# Load Dataset
data_path = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
nba_data = spark.read.csv(data_path, header=True, inferSchema=True)
print(f"Dataset loaded: {nba_data.count()} rows, {len(nba_data.columns)} columns")

# =========================================================
# Folder Setup for Results
# =========================================================
results_dir = f"/home/sat3812/BD_Project/Results/RandomForest"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# =========================================================
# Feature Preparation
# =========================================================
target = [col for col in nba_data.columns if "PER" in col.upper()][0]
print(f"Target variable selected: {target}")

feature_columns = [
    col for col, dtype in nba_data.dtypes
    if dtype in ("int", "double", "float", "bigint") and col != target
]

assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
data_with_features = assembler.transform(nba_data.na.drop(subset=[target]))

scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
scaled_data = scaler.fit(data_with_features).transform(data_with_features)

# =========================================================
# Binary Label Creation (Based on Average PER)
# =========================================================
average_per = nba_data.select(F.mean(F.col(target))).collect()[0][0]
print(f"Average PER (classification threshold): {average_per:.2f}")

labeled_data = nba_data.withColumn(
    "label",
    F.when(F.col(target) >= average_per, 1).otherwise(0)
)

data_with_features = assembler.transform(labeled_data.na.drop(subset=[target]))
scaled_classification_data = scaler.fit(data_with_features).transform(data_with_features).select("features", "label")
train_data, test_data = scaled_classification_data.randomSplit([0.8, 0.2], seed=42)
print(f"Train set: {train_data.count()} | Test set: {test_data.count()}")

# =========================================================
# Visualize PER Distribution with Threshold
# =========================================================
per_distribution = nba_data.select(target).toPandas()
plt.figure(figsize=(9, 5))
sns.histplot(per_distribution[target], bins=35, kde=True, edgecolor='black', color='teal')
plt.axvline(average_per, linestyle='--', linewidth=2, color='red', label=f'Threshold = {average_per:.2f}')
plt.title("Player Efficiency Rating (PER) Distribution", fontsize=14, weight='bold')
plt.xlabel("PER Value"); plt.ylabel("Player Count"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "PER_Distribution.png"))
plt.close()

# =========================================================
# Train Random Forest Classifier
# =========================================================
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    predictionCol="prediction",
    numTrees=200,
    maxDepth=12,
    seed=42,
    featureSubsetStrategy="auto"
)

start_time = time.time()
rf_model = rf.fit(train_data)
predictions = rf_model.transform(test_data).cache()
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# =========================================================
# Evaluate Model Performance
# =========================================================
accuracy_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
precision_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_eval    = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
f1_eval        = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
auc_eval       = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

results_df = pd.DataFrame({
    "Model": ["Random Forest Classifier"],
    "Accuracy":  [accuracy_eval.evaluate(predictions)],
    "Precision": [precision_eval.evaluate(predictions)],
    "Recall":    [recall_eval.evaluate(predictions)],
    "F1":        [f1_eval.evaluate(predictions)],
    "AUC":       [auc_eval.evaluate(predictions)],
    "Training_Time(s)": [training_time]
})

print("\nClassification Performance:")
print(results_df.round(3))

# Save metrics
metrics_path = os.path.join(results_dir, "RF_Classification_Results.csv")
results_df.to_csv(metrics_path, index=False)
print(f"Results saved to: {metrics_path}")

# =========================================================
# Confusion Matrix Visualization
# =========================================================
preds_pd = predictions.select("label", "prediction").toPandas()
cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Below Avg", "Above/Equal Avg"])
disp.plot(cmap="Greens")
plt.title("Random Forest Classification Confusion Matrix", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Confusion_Matrix.png"))
plt.close()

# =========================================================
# ROC & Precision–Recall Curves
# =========================================================
prob_pd = predictions.select("label", "probability").toPandas()
pos_scores = prob_pd["probability"].apply(lambda v: float(v[1]))

fpr, tpr, _ = roc_curve(prob_pd["label"], pos_scores)
prec, rec, _ = precision_recall_curve(prob_pd["label"], pos_scores)
roc_auc = roc_auc_score(prob_pd["label"], pos_scores)
pr_auc  = average_precision_score(prob_pd["label"], pos_scores)

# ROC Curve
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, lw=2, color='blue', label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--", color='gray')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Random Forest"); plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "ROC_Curve.png"))
plt.close()

# Precision–Recall Curve
plt.figure(figsize=(7,5))
plt.plot(rec, prec, lw=2, color='darkorange', label=f"AP = {pr_auc:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve — Random Forest")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "Precision_Recall_Curve.png"))
plt.close()

