# =========================================================
# Logistic Regression 
# =========================================================
# Dependencies and Environment Setup
# =========================================================
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as spark_fn
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

# =========================================================
# Configuration and Hyperparameters
# =========================================================
NBA_DATA_PATH = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
RESULTS_OUTPUT_PATH = "/content/"
TRAIN_TEST_SPLIT = [0.8, 0.2]
SPLIT_SEED_VALUE = 42
LR_MAX_ITERATIONS = 150
LR_REG_PARAMETER = 0.01
LR_ELASTIC_NET_PARAM = 0.2
CLASSIFICATION_THRESHOLD = 0.5

# =========================================================
# Initialize Spark Session
# =========================================================
def establish_spark_environment():
    """Create Spark session for distributed computing"""
    return SparkSession.builder \
        .appName("NBA_LR_Classification") \
        .getOrCreate()

spark_engine = establish_spark_environment()
print("✓ Spark environment initialized")

# =========================================================
# Data Loading and Exploration
# =========================================================
def import_basketball_dataset(file_location):
    """Load NBA player statistics dataset"""
    dataset = spark_engine.read.csv(file_location, header=True, inferSchema=True)
    total_rows = dataset.count()
    total_columns = len(dataset.columns)
    print(f"Loading dataset...")
    print(f"✓ Loaded {total_rows} rows with {total_columns} columns")
    return dataset

nba_dataset = import_basketball_dataset(NBA_DATA_PATH)

# =========================================================
# Output Directory Management
# =========================================================
def ensure_output_directory(directory_path):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    print(f"Results will be saved to: {directory_path}")

ensure_output_directory(RESULTS_OUTPUT_PATH)

# =========================================================
# Target Variable and Feature Engineering
# =========================================================
def locate_efficiency_column(dataframe):
    """Find Player Efficiency Rating (PER) column"""
    per_candidates = [col for col in dataframe.columns if "PER" in col.upper()]
    if not per_candidates:
        raise ValueError("❌ Target variable (PER) not found.")
    efficiency_col = per_candidates[0]
    print(f"Using {efficiency_col} as target metric")
    return efficiency_col

target_metric = locate_efficiency_column(nba_dataset)

def calculate_performance_threshold(dataframe, metric_col):
    """Compute mean efficiency as binary classification threshold"""
    avg_metric = dataframe.select(spark_fn.mean(spark_fn.col(metric_col))).collect()[0][0]
    print(f"Average {metric_col}: {avg_metric:.2f}")
    return avg_metric

efficiency_threshold = calculate_performance_threshold(nba_dataset, target_metric)

def extract_quantitative_columns(dataframe, exclude_col):
    """Extract all numeric feature columns"""
    numeric_cols = [
        col for col, dtype in dataframe.dtypes
        if dtype in ("int", "bigint", "float", "double") and col != exclude_col
    ]
    print(f"Using {len(numeric_cols)} numeric features")
    return numeric_cols

feature_collection = extract_quantitative_columns(nba_dataset, target_metric)

# =========================================================
# Create Binary Classification Labels
# =========================================================
def generate_binary_classification_labels(dataframe, target_col, threshold_value):
    """Create binary labels: 1 = above average, 0 = below average"""
    return dataframe.withColumn(
        "label",
        spark_fn.when(spark_fn.col(target_col) >= threshold_value, 1).otherwise(0)
    )

labeled_dataset = generate_binary_classification_labels(nba_dataset, target_metric, efficiency_threshold)

# =========================================================
# Feature Preprocessing and Scaling Pipeline
# =========================================================
def assemble_feature_vectors(dataframe, feature_cols, exclude_col):
    """Combine numeric columns into single feature vector"""
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="raw_features",
        handleInvalid="skip"
    )
    return assembler.transform(dataframe)

def normalize_feature_scale(dataframe):
    """Apply min-max normalization to feature vectors"""
    scaler = MinMaxScaler(inputCol="raw_features", outputCol="features")
    scaling_model = scaler.fit(dataframe)
    return scaling_model.transform(dataframe)

# Apply preprocessing pipeline
vectorized_data = assemble_feature_vectors(labeled_dataset, feature_collection, target_metric)
scaled_data = normalize_feature_scale(vectorized_data)
prepared_dataset = scaled_data.select("features", "label")

# =========================================================
# Train-Test Data Partitioning
# =========================================================
def partition_dataset(dataframe, split_ratios, random_seed):
    """Divide data into training and testing subsets"""
    training_partition, testing_partition = dataframe.randomSplit(split_ratios, seed=random_seed)
    train_count = training_partition.count()
    test_count = testing_partition.count()
    print(f"Training samples: {train_count} | Testing samples: {test_count}")
    return training_partition, testing_partition

train_subset, test_subset = partition_dataset(prepared_dataset, TRAIN_TEST_SPLIT, SPLIT_SEED_VALUE)

# =========================================================
# Logistic Regression Model Training
# =========================================================
def train_logistic_model(train_data, max_iter, reg_param, elastic_param, thresh):
    """Train Logistic Regression classifier"""
    print("Training Logistic Regression model...")
    
    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_param,
        threshold=thresh
    )
    
    training_start = time.time()
    trained_model = classifier.fit(train_data)
    training_elapsed = time.time() - training_start
    
    print(f"✓ Training completed in {training_elapsed:.2f}s")
    return trained_model, training_elapsed

lr_model, lr_training_time = train_logistic_model(
    train_subset,
    LR_MAX_ITERATIONS,
    LR_REG_PARAMETER,
    LR_ELASTIC_NET_PARAM,
    CLASSIFICATION_THRESHOLD
)

# =========================================================
# Generate Predictions on Test Set
# =========================================================
def apply_model_predictions(model, test_data):
    """Apply trained model to generate predictions"""
    print("Generating predictions...")
    prediction_start = time.time()
    predictions = model.transform(test_data).cache()
    prediction_elapsed = time.time() - prediction_start
    print(f"✓ Predictions completed in {prediction_elapsed:.2f}s")
    return predictions, prediction_elapsed

predictions_data, lr_prediction_time = apply_model_predictions(lr_model, test_subset)

# =========================================================
# Model Performance Evaluation
# =========================================================
def compute_classification_metrics(predictions_df):
    """Calculate comprehensive classification performance metrics"""
    evaluators = {
        "accuracy": MulticlassClassificationEvaluator(metricName="accuracy"),
        "weightedPrecision": MulticlassClassificationEvaluator(metricName="weightedPrecision"),
        "weightedRecall": MulticlassClassificationEvaluator(metricName="weightedRecall"),
        "f1": MulticlassClassificationEvaluator(metricName="f1"),
        "areaUnderROC": BinaryClassificationEvaluator(metricName="areaUnderROC"),
        "areaUnderPR": BinaryClassificationEvaluator(metricName="areaUnderPR")
    }
    
    metrics_computed = {
        metric_name: evaluator.evaluate(predictions_df)
        for metric_name, evaluator in evaluators.items()
    }
    
    return metrics_computed

model_metrics = compute_classification_metrics(predictions_data)

# =========================================================
# Results Summary and Display
# =========================================================
def format_results_dataframe(metrics_dict, train_time, pred_time):
    """Compile results into structured format"""
    results_summary = pd.DataFrame({
        "Model": ["Logistic Regression Classifier"],
        "Accuracy": [metrics_dict["accuracy"]],
        "Precision": [metrics_dict["weightedPrecision"]],
        "Recall": [metrics_dict["weightedRecall"]],
        "F1-Score": [metrics_dict["f1"]],
        "ROC-AUC": [metrics_dict["areaUnderROC"]],
        "PR-AUC": [metrics_dict["areaUnderPR"]],
        "Train_Time(s)": [train_time],
        "Predict_Time(s)": [pred_time]
    })
    return results_summary

results_table = format_results_dataframe(model_metrics, lr_training_time, lr_prediction_time)

print("\nClassification Performance Summary:")
print(results_table.to_string(index=False, justify='center'))
results_table.to_csv(
    os.path.join(RESULTS_OUTPUT_PATH, "LogisticRegression_Results.csv"),
    index=False
)

# =========================================================
# Confusion Matrix Visualization
# =========================================================
def visualize_confusion_matrix(predictions_df, output_dir):
    """Generate and save confusion matrix plot"""
    predictions_pandas = predictions_df.select("label", "prediction").toPandas()
    
    conf_matrix = confusion_matrix(predictions_pandas["label"], predictions_pandas["prediction"])
    matrix_display = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=["Below Avg", "Above Avg"]
    )
    
    matrix_display.plot(cmap="Blues")
    plt.title("Logistic Regression Confusion Matrix", fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "Confusion_Matrix_LR.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Confusion matrix saved: {output_file}")

visualize_confusion_matrix(predictions_data, RESULTS_OUTPUT_PATH)

# =========================================================
# ROC Curve Generation
# =========================================================
def plot_roc_curve(predictions_df, output_dir):
    """Generate and save ROC curve"""
    probability_data = predictions_df.select("label", "probability").toPandas()
    probability_data["prob_positive_class"] = probability_data["probability"].apply(
        lambda v: float(v[1])
    )
    
    fpr, tpr, _ = roc_curve(probability_data["label"], probability_data["prob_positive_class"])
    roc_auc_score_value = roc_auc_score(probability_data["label"], probability_data["prob_positive_class"])
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, color='darkgreen', label=f"AUC = {roc_auc_score_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Logistic Regression", fontsize=14, weight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "ROC_Curve.png")
    plt.savefig(output_file)
    plt.close()
    print(f"ROC curve saved: {output_file}")

plot_roc_curve(predictions_data, RESULTS_OUTPUT_PATH)

# =========================================================
# Precision-Recall Curve Generation
# =========================================================
def plot_precision_recall_curve(predictions_df, output_dir):
    """Generate and save precision-recall curve"""
    probability_data = predictions_df.select("label", "probability").toPandas()
    probability_data["prob_positive_class"] = probability_data["probability"].apply(
        lambda v: float(v[1])
    )
    
    precision_vals, recall_vals, _ = precision_recall_curve(
        probability_data["label"],
        probability_data["prob_positive_class"]
    )
    pr_auc_score_value = average_precision_score(
        probability_data["label"],
        probability_data["prob_positive_class"]
    )
    
    plt.figure(figsize=(7, 5))
    plt.plot(recall_vals, precision_vals, lw=2, color='darkorange',
             label=f"AP = {pr_auc_score_value:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — Logistic Regression", fontsize=14, weight='bold')
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, "Precision_Recall_Curve_LR.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Precision-recall curve saved: {output_file}")

plot_precision_recall_curve(predictions_data, RESULTS_OUTPUT_PATH)

# =========================================================
# Save Model Predictions
# =========================================================
def export_predictions_to_csv(predictions_df, output_dir):
    """Extract and save prediction results to CSV"""
    from pyspark.sql.types import DoubleType
    
    # Extract probability of positive class
    extract_positive_prob = spark_fn.udf(
        lambda v: float(v[1]) if v is not None else None,
        DoubleType()
    )
    
    predictions_with_prob = predictions_df.withColumn(
        "Prob_Class1",
        extract_positive_prob(spark_fn.col("probability"))
    )
    
    # Select final columns
    final_output = predictions_with_prob.select(
        spark_fn.col("label").alias("True_Label"),
        spark_fn.col("prediction").alias("Predicted_Label"),
        spark_fn.col("Prob_Class1").alias("Probability_Class_1")
    )
    
    # Save clean CSV
    output_path = os.path.join(output_dir, "Predictions_LR")
    final_output.write.mode("overwrite").csv(output_path, header=True)
    print(f"Predictions exported: {output_path}")

export_predictions_to_csv(predictions_data, RESULTS_OUTPUT_PATH)

# =========================================================
# Pipeline Completion Summary
# =========================================================
print("\n" + "="*60)
print("Logistic Regression Classification Pipeline Complete")
print("="*60)
print(f"Total Execution Time:")
print(f"  - Training: {lr_training_time:.2f}s")
print(f"  - Prediction: {lr_prediction_time:.2f}s")
print("="*60)
