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

# Start Spark Session
spark = SparkSession.builder.appName("NBA_RF_Classification").getOrCreate()

# Load Dataset
data_path = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
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
# Visualize PER Distribution with Threshold
# =========================================================
per_distribution = nba_data.select(target).toPandas()
plt.figure(figsize=(9, 5))
sns.histplot(per_distribution[target], bins=35, kde=True, edgecolor='black')
plt.axvline(average_per, linestyle='--', linewidth=2, label=f'Threshold = {average_per:.2f}')
plt.title("Player Efficiency Rating (PER) Distribution", fontsize=14, weight='bold')
plt.xlabel("PER Value", fontsize=12)
plt.ylabel("Player Count", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

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
# Evaluate Model Performance (Accuracy / Precision / Recall / F1 / AUC)
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
    "AUC":       [auc_eval.evaluate(predictions)]
})

print("\nClassification Performance:")
try:
    display(
        results_df.style
        .background_gradient(cmap="Greens")
        .format(subset=["Accuracy", "Precision", "Recall", "F1", "AUC"], formatter="{:.3f}")
    )
except NameError:
    # Fallback if display(...) is not available
    print(results_df.round(3))

# =========================================================
# Confusion Matrix Visualization
# =========================================================
preds_pd = predictions.select("label", "prediction").toPandas()
cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Below Avg", "Above/Equal Avg"])
disp.plot(cmap="Greens")
plt.title("Random Forest Classification Confusion Matrix", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

# =========================================================
# ROC & Precision–Recall Curves (Binary)
# =========================================================
# Extract probability of the positive class (label==1)
prob_pd = predictions.select("label", "probability").toPandas()
pos_scores = prob_pd["probability"].apply(lambda v: float(v[1]))  # probability for class 1

fpr, tpr, _ = roc_curve(prob_pd["label"], pos_scores)
prec, rec, _ = precision_recall_curve(prob_pd["label"], pos_scores)
roc_auc = roc_auc_score(prob_pd["label"], pos_scores)
pr_auc  = average_precision_score(prob_pd["label"], pos_scores)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Random Forest")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,5))
plt.plot(rec, prec, lw=2, label=f"AP = {pr_auc:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve — Random Forest")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()


