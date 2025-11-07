
# Spark MLP Classification Model for NBA Player Efficiency
# Import Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import os

# Start Spark Session
spark = SparkSession.builder.appName("NBA_MLP_Classification").getOrCreate()

# Load Dataset
data_path = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
nba_data = spark.read.csv(data_path, header=True, inferSchema=True)
print(f"Dataset loaded: {nba_data.count()} rows, {len(nba_data.columns)} columns")
# Create results folder
results_dir = "/home/hduser/BigDataProject/Results"
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved in: {results_dir}")

# Feature Preparation
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

# Standardize features
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
scaled_data = scaler.fit(data_with_features).transform(data_with_features)

# Binary Label Creation (Based on Average PER)
# Compute average PER and use it as threshold
average_per = nba_data.select(F.mean(F.col(target))).collect()[0][0]
print(f"Average PER (classification threshold): {average_per:.2f}")

# Create binary label column (Above Average = 1, Below Average = 0)
labeled_data = nba_data.withColumn(
    "label",
    F.when(F.col(target) >= average_per, 1).otherwise(0)
)

# Reassemble features for classification
data_with_features = assembler.transform(labeled_data)
scaled_classification_data = scaler.fit(data_with_features).transform(data_with_features).select("features", "label")

# Split dataset into training and testing
train_data, test_data = scaled_classification_data.randomSplit([0.8, 0.2], seed=42)
print(f"Train set: {train_data.count()} | Test set: {test_data.count()}")

# Visualize PER Distribution with Threshold
per_distribution = nba_data.select(target).toPandas()
plt.figure(figsize=(9, 5))
sns.histplot(per_distribution[target], bins=35, kde=True, color='darkorange', edgecolor='black')
plt.axvline(average_per, color='navy', linestyle='--', linewidth=2, label=f'Threshold = {average_per:.2f}')
plt.title("Player Efficiency Rating (PER) Distribution", fontsize=14, weight='bold')
plt.xlabel("PER Value", fontsize=12)
plt.ylabel("Player Count", fontsize=12)
plt.legend()
plt.tight_layout()
per_plot_path = os.path.join(results_dir, "PER_Distribution.png")
plt.savefig(per_plot_path, dpi=300)
plt.close()
print(f"PER distribution plot saved at: {per_plot_path}")
# Train MLP Neural Network Classifier
network_layers = [len(feature_columns), 20, 10, 2]
mlp_classifier = MultilayerPerceptronClassifier(
    featuresCol="features",
    labelCol="label",
    layers=network_layers,
    maxIter=100
)

start_time = time.time()
trained_mlp = mlp_classifier.fit(train_data)
predictions = trained_mlp.transform(test_data)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Evaluate Model Performance
accuracy_eval = MulticlassClassificationEvaluator(metricName="accuracy")
precision_eval = MulticlassClassificationEvaluator(metricName="weightedPrecision")
recall_eval = MulticlassClassificationEvaluator(metricName="weightedRecall")
auc_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")

results_df = pd.DataFrame({
    "Model": ["Multilayer Perceptron Classifier"],
    "Accuracy": [accuracy_eval.evaluate(predictions)],
    "Precision": [precision_eval.evaluate(predictions)],
    "Recall": [recall_eval.evaluate(predictions)],
    "AUC": [auc_eval.evaluate(predictions)]
})

print("\nClassification Performance:")
display(
    results_df.style
    .background_gradient(cmap="Purples")
    .format(subset=["Accuracy", "Precision", "Recall", "AUC"], formatter="{:.3f}")
)

# Confusion Matrix Visualization
preds_pd = predictions.select("label", "prediction").toPandas()
cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Below Avg", "Above Avg"])
disp.plot(cmap="Oranges")
plt.title("MLP Classification Confusion Matrix", fontsize=14, weight='bold')
plt.tight_layout()
cm_plot_path = os.path.join(results_dir, "Confusion_Matrix.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"Confusion matrix saved at: {cm_plot_path}")
