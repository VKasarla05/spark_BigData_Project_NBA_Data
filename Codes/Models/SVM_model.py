#Import Libraries
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

# Start Spark Session
spark = SparkSession.builder.appName("NBA_RF_Classification").getOrCreate()

# Load Dataset
data_path = "/content/part-*.csv"
nba_data = spark.read.csv(data_path, header=True, inferSchema=True)
print(f"Dataset loaded: {nba_data.count()} rows, {len(nba_data.columns)} columns")
# =========================================================
# Feature Preparation
# =========================================================

# Identify target variable containing "PER"
target = [col for col in nba_data.columns if "PER" in col.upper()][0]
print(f"Target variable selected: {target}")

# Select numeric feature columns
feature_columns = [
    col for col, dtype in nba_data.dtypes
    if dtype in ("int", "double", "float", "bigint") and col != target
]

# Combine numeric columns into a single feature vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
data_with_features = assembler.transform(nba_data.na.drop(subset=[target]))

# Standardize features (not strictly needed for RF, but kept to mirror your style)
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
scaled_data = scaler.fit(data_with_features).transform(data_with_features)

# =========================================================
# Binary Label Creation (Based on Average PER)
# =========================================================

# Compute average PER and use it as threshold
average_per = nba_data.select(F.mean(F.col(target))).collect()[0][0]
print(f"Average PER (classification threshold): {average_per:.2f}")

# Create binary label column (Above/Equal Avg = 1, Below Avg = 0)
labeled_data = nba_data.withColumn(
    "label",
    F.when(F.col(target) >= average_per, 1).otherwise(0)
)

# Reassemble features for classification
data_with_features = assembler.transform(labeled_data.na.drop(subset=[target]))
scaled_classification_data = scaler.fit(data_with_features).transform(data_with_features).select("features", "label")

# Split dataset into training and testing
train_data, test_data = scaled_classification_data.randomSplit([0.8, 0.2], seed=42)
print(f"Train set: {train_data.count()} | Test set: {test_data.count()}")

# =========================================================
# (Optional) Visualize PER Distribution with Threshold (if using numeric PER)
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
# =============================================================================
  # Model Building SVM
  # =========================================================
# SVM Model Training and Evaluation
# =========================================================

from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Linear Support Vector Classifier
svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=100, regParam=0.1)

# Train the model
svm_model = svm.fit(train_data)

# Make predictions on test set
predictions = svm_model.transform(test_data)

# =========================================================
# Evaluation Metrics
# =========================================================

# Initialize evaluators
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)

# Compute metrics
accuracy = accuracy_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

print("==============================================")
print("SVM Classification Results")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print("==============================================")

# =========================================================
# Model Summary
# =========================================================

# Show a few predictions
predictions.select("label", "prediction", "features").show(10, truncate=False)

# Retrieve model coefficients and intercept
print("\nModel Coefficients (first 10):")
print(svm_model.coefficients[:10])
print(f"Intercept: {svm_model.intercept}")

# =========================================================
# Save Model (Optional)
# =========================================================

# Save the trained SVM model
svm_model.save("clean_output/svm_nba_model")

print("Model saved successfully to 'clean_output/svm_nba_model'")
# =========================================================
# Visualize SVM Model Performance
# =========================================================

import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

# Convert predictions to Pandas DataFrame for visualization
preds_pd = predictions.select("label", "prediction").toPandas()

# =========================================================
# Confusion Matrix
# =========================================================
cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])
cm_df = pd.DataFrame(cm, index=["Actual 0 (Below Avg)", "Actual 1 (Above Avg)"],
                        columns=["Predicted 0", "Predicted 1"])

plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix – NBA PER Classification")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("clean_output/svm_confusion_matrix.png")
plt.show()

# =========================================================
# Performance Metrics Calculation
# =========================================================
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="precisionByLabel"
)
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="recallByLabel"
)

precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)
accuracy = accuracy_evaluator.evaluate(predictions)
f1 = f1_evaluator.evaluate(predictions)

# =========================================================
# Performance Metrics Visualization
# =========================================================
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

plt.figure(figsize=(7, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
plt.title("SVM Classification Performance Metrics – NBA PER Prediction")
plt.ylim(0, 1)
for i, v in enumerate(list(metrics.values())):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("clean_output/svm_performance_metrics.png")
plt.show()

# =========================================================
# Classification Report (Optional)
# =========================================================
print("\nClassification Report:")
print(classification_report(preds_pd["label"], preds_pd["prediction"], digits=3))
