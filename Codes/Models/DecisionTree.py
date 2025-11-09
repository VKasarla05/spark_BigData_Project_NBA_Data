# --------------Decision Tree----------------------
# Modules Importing
import os
import time as elapsed_duration_tracker
from pyspark.sql import SparkSession as DistributedComputeEngine
from pyspark.sql import functions as tree_operations
from pyspark.ml.feature import VectorAssembler as AttributeVectorCombiner
from pyspark.ml.feature import StandardScaler as FeatureValueNormalizer
from pyspark.ml.classification import DecisionTreeClassifier as HierarchicalTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as MultiLabelMetricsEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator as BinaryMetricsEvaluator
from sklearn.metrics import (
    confusion_matrix as compute_error_matrix,
    ConfusionMatrixDisplay as ErrorMatrixVisualizer,
    roc_curve as receiver_operator_curve_calculation,
    precision_recall_curve as precision_recall_tradeoff_curve,
    roc_auc_score as calculate_roc_area,
    average_precision_score as calculate_average_precision,
    precision_recall_fscore_support as comprehensive_metric_calculator,
    classification_report as detailed_classification_summary
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ATHLETE_STATISTICS_FILE_PATTERN = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
PERFORMANCE_OUTPUT_DIRECTORY = "/content/"
TRAINING_DATASET_FRACTION = 0.8
TESTING_DATASET_FRACTION = 0.2
RANDOM_REPRODUCTION_SEED = 42
TREE_MAXIMUM_DEPTH = 12
TREE_MINIMUM_INSTANCES = 1
TREE_MINIMUM_INFO_GAIN = 0.0
TREE_IMPURITY_METRIC = "gini"
TARGET_METRIC_CANDIDATES = ["ss_PER_quantile", "ss_PER", "PER"]

# Distributed Computing Environment Initialization
distributed_compute_engine = DistributedComputeEngine.builder.appName(
    "DecisionTree_Athletic_Classification"
).getOrCreate()
print("Distributed computing environment initialized successfully")

# Athletic Performance Dataset Loading
athlete_performance_dataset = distributed_compute_engine.read.csv(
    ATHLETE_STATISTICS_FILE_PATTERN,
    header=True,
    inferSchema=True
)
dataset_row_count = athlete_performance_dataset.count()
dataset_column_count = len(athlete_performance_dataset.columns)
print(f"Athletic dataset acquisition complete: {dataset_row_count} records, {dataset_column_count} attributes")

# Output Storage
os.makedirs(PERFORMANCE_OUTPUT_DIRECTORY, exist_ok=True)
print(f"Performance artifacts directory: {PERFORMANCE_OUTPUT_DIRECTORY}")

# Target Metric Identification and Validation
identified_target_metric = next(
    (metric_name for metric_name in TARGET_METRIC_CANDIDATES 
     if metric_name in athlete_performance_dataset.columns),
    None
)
if identified_target_metric is None:
    raise ValueError(
        f"No target metric found. Expected one of: {', '.join(TARGET_METRIC_CANDIDATES)}"
    )

print(f"Target performance metric selected: {identified_target_metric}")

# Numerical Attribute Extraction and Selection
extracted_feature_columns = [
    column_identifier for column_identifier, column_datatype in athlete_performance_dataset.dtypes
    if column_datatype in ("int", "double", "float", "bigint") 
    and column_identifier != identified_target_metric
]
print(f"Numerical features extracted: {len(extracted_feature_columns)}")

# Feature Vector Construction
attribute_combining_transformer = AttributeVectorCombiner(
    inputCols=extracted_feature_columns,
    outputCol="raw_features"
)
dataset_with_combined_vectors = attribute_combining_transformer.transform(
    athlete_performance_dataset.na.drop(subset=[identified_target_metric])
)

# Target Label Preparation
if identified_target_metric == "ss_PER_quantile":
    dataset_with_labels = dataset_with_combined_vectors.withColumn(
        "label",
        tree_operations.col(identified_target_metric).cast("int")
    )
    label_preparation_note = "Using quantile bins as multiclass classification targets"
    efficiency_threshold_value = None
else:
    efficiency_threshold_value = athlete_performance_dataset.select(
        tree_operations.mean(tree_operations.col(identified_target_metric))
    ).collect()[0][0]
    dataset_with_labels = dataset_with_combined_vectors.withColumn(
        "label",
        tree_operations.when(
            tree_operations.col(identified_target_metric) >= efficiency_threshold_value, 
            1
        ).otherwise(0)
    )
    label_preparation_note = f"Binary classification using threshold: {efficiency_threshold_value:.4f}"

print(f"Label preparation strategy: {label_preparation_note}")

# Feature Normalization and Scaling
feature_normalization_transformer = FeatureValueNormalizer(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)
normalization_model_fitted = feature_normalization_transformer.fit(dataset_with_labels)
dataset_with_normalized_features = normalization_model_fitted.transform(dataset_with_labels)

model_ready_dataset = dataset_with_normalized_features.select("features", "label")

# Training and Testing Dataset Partitioning
training_dataset_partition, testing_dataset_partition = model_ready_dataset.randomSplit(
    [TRAINING_DATASET_FRACTION, TESTING_DATASET_FRACTION],
    seed=RANDOM_REPRODUCTION_SEED
)
training_dataset_size = training_dataset_partition.count()
testing_dataset_size = testing_dataset_partition.count()
print(f"Data partitioning: {training_dataset_size} training | {testing_dataset_size} testing")

# Performance Metric Distribution Visualization
if efficiency_threshold_value is not None:
    performance_metric_values = athlete_performance_dataset.select(identified_target_metric).toPandas()
    
    plt.figure(figsize=(9, 5))
    sns.histplot(
        performance_metric_values[identified_target_metric],
        bins=35,
        kde=True,
        edgecolor='black',
        color='royalblue'
    )
    plt.axvline(
        efficiency_threshold_value,
        linestyle='--',
        linewidth=2,
        color='red',
        label=f'Threshold = {efficiency_threshold_value:.2f}'
    )
    plt.title(f"{identified_target_metric} Distribution Analysis", fontsize=14, weight='bold')
    plt.xlabel(identified_target_metric)
    plt.ylabel("Observation Count")
    plt.legend()
    plt.tight_layout()
    
    distribution_visualization_path = os.path.join(PERFORMANCE_OUTPUT_DIRECTORY, "PER_Distribution_DT.png")
    plt.savefig(distribution_visualization_path)
    plt.close()
    print(f"Distribution visualization saved: {distribution_visualization_path}")

# Decision Tree Model Configuration and Training
hierarchical_tree_classifier = HierarchicalTreeClassifier(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    maxDepth=TREE_MAXIMUM_DEPTH,
    minInstancesPerNode=TREE_MINIMUM_INSTANCES,
    minInfoGain=TREE_MINIMUM_INFO_GAIN,
    impurity=TREE_IMPURITY_METRIC,
    seed=RANDOM_REPRODUCTION_SEED
)

print("Initiating decision tree model training...")
training_execution_start_time = elapsed_duration_tracker.time()
fitted_tree_model = hierarchical_tree_classifier.fit(training_dataset_partition)
total_training_elapsed_time = elapsed_duration_tracker.time() - training_execution_start_time
print(f"Model training completed: {total_training_elapsed_time:.2f} seconds")


# Model Predictions Generation
test_dataset_predictions = fitted_tree_model.transform(testing_dataset_partition).cache()


# Performance Metrics Calculation
accuracy_metric_evaluator = MultiLabelMetricsEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
f1_score_metric_evaluator = MultiLabelMetricsEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
weighted_precision_metric_evaluator = MultiLabelMetricsEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
weighted_recall_metric_evaluator = MultiLabelMetricsEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

computed_accuracy_metric = accuracy_metric_evaluator.evaluate(test_dataset_predictions)
computed_f1_metric = f1_score_metric_evaluator.evaluate(test_dataset_predictions)
computed_precision_metric = weighted_precision_metric_evaluator.evaluate(test_dataset_predictions)
computed_recall_metric = weighted_recall_metric_evaluator.evaluate(test_dataset_predictions)

# Area Under Curve Calculation (ROC)
try:
    roc_area_evaluator = BinaryMetricsEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    computed_auc_metric = roc_area_evaluator.evaluate(test_dataset_predictions)
except Exception as e:
    print(f"ROC AUC calculation skipped: {e}")
    computed_auc_metric = float('nan')

# Results
performance_results_table = pd.DataFrame({
    "Model": ["Decision Tree Hierarchical Classifier"],
    "Accuracy": [computed_accuracy_metric],
    "Precision": [computed_precision_metric],
    "Recall": [computed_recall_metric],
    "F1": [computed_f1_metric],
    "AUC": [computed_auc_metric],
    "Training_Time(s)": [total_training_elapsed_time]
})

print("\nPerformance Metrics Summary:")
print(performance_results_table.to_string(index=False, justify='center'))

results_csv_output_path = os.path.join(PERFORMANCE_OUTPUT_DIRECTORY, "DT_Classification_Results.csv")
performance_results_table.to_csv(results_csv_output_path, index=False)
print(f"Results exported: {results_csv_output_path}")

# Confusion Matrix Generation and Visualization
predictions_with_labels = test_dataset_predictions.select("label", "prediction").toPandas()
computed_confusion_matrix = compute_error_matrix(
    predictions_with_labels["label"],
    predictions_with_labels["prediction"]
)

confusion_matrix_visualizer = ErrorMatrixVisualizer(
    confusion_matrix=computed_confusion_matrix
)
confusion_matrix_visualizer.plot(cmap="Blues")
plt.title("Decision Tree Classification — Confusion Matrix", fontsize=14, weight='bold')
plt.tight_layout()

confusion_matrix_output_path = os.path.join(PERFORMANCE_OUTPUT_DIRECTORY, "Confusion_Matrix_DT.png")
plt.savefig(confusion_matrix_output_path)
plt.close()
print(f"Confusion matrix visualization saved: {confusion_matrix_output_path}")

# ROC Curve Visualization
probability_predictions_data = test_dataset_predictions.select("label", "probability").toPandas()
try:
    extracted_positive_class_scores = probability_predictions_data["probability"].apply(
        lambda probability_vector: float(probability_vector[1])
    )
    
    calculated_roc_auc_value = calculate_roc_area(
        probability_predictions_data["label"],
        extracted_positive_class_scores
    )
    calculated_pr_auc_value = calculate_average_precision(
        probability_predictions_data["label"],
        extracted_positive_class_scores
    )
    
    fpr_coordinates, tpr_coordinates, _ = receiver_operator_curve_calculation(
        probability_predictions_data["label"],
        extracted_positive_class_scores
    )
    precision_coordinates, recall_coordinates, _ = precision_recall_tradeoff_curve(
        probability_predictions_data["label"],
        extracted_positive_class_scores
    )
    
    # ROC Curve Plot
    plt.figure(figsize=(7, 5))
    plt.plot(
        fpr_coordinates,
        tpr_coordinates,
        lw=2,
        color='blue',
        label=f"AUC = {calculated_roc_auc_value:.3f}"
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Decision Tree Classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    roc_output_path = os.path.join(PERFORMANCE_OUTPUT_DIRECTORY, "ROC_Curve_DT.png")
    plt.savefig(roc_output_path)
    plt.close()
    print(f"ROC curve visualization saved: {roc_output_path}")
    
    # Precision-Recall Curve Plot
    plt.figure(figsize=(7, 5))
    plt.plot(
        recall_coordinates,
        precision_coordinates,
        lw=2,
        color='orange',
        label=f"AP = {calculated_pr_auc_value:.3f}"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve — Decision Tree Classifier")
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    pr_output_path = os.path.join(PERFORMANCE_OUTPUT_DIRECTORY, "Precision_Recall_Curve_DT.png")
    plt.savefig(pr_output_path)
    plt.close()
    print(f"Precision-recall curve visualization saved: {pr_output_path}")
    
except Exception as exception_details:
    print(f"ROC and PR curve generation skipped: {exception_details}")
    
# Comprehensive Detailed Metrics Computation
true_labels_array = predictions_with_labels["label"].values.astype(int)
predicted_labels_array = predictions_with_labels["prediction"].values.astype(int)

macro_precision_avg, macro_recall_avg, macro_f1_avg, _ = comprehensive_metric_calculator(
    true_labels_array,
    predicted_labels_array,
    average='macro',
    zero_division=0
)
micro_precision_avg, micro_recall_avg, micro_f1_avg, _ = comprehensive_metric_calculator(
    true_labels_array,
    predicted_labels_array,
    average='micro',
    zero_division=0
)
weighted_precision_avg, weighted_recall_avg, weighted_f1_avg, _ = comprehensive_metric_calculator(
    true_labels_array,
    predicted_labels_array,
    average='weighted',
    zero_division=0
)

print("\n" + "="*60)
print("DETAILED CLASSIFICATION METRICS")
print("="*60)
print(f"Macro Averaging     → Precision: {macro_precision_avg:.4f} | Recall: {macro_recall_avg:.4f} | F1: {macro_f1_avg:.4f}")
print(f"Micro Averaging     → Precision: {micro_precision_avg:.4f} | Recall: {micro_recall_avg:.4f} | F1: {micro_f1_avg:.4f}")
print(f"Weighted Averaging  → Precision: {weighted_precision_avg:.4f} | Recall: {weighted_recall_avg:.4f} | F1: {weighted_f1_avg:.4f}")

detailed_classification_report = detailed_classification_summary(
    true_labels_array,
    predicted_labels_array,
    digits=4,
    zero_division=0
)
print("\nDetailed Classification Report:")
print(detailed_classification_report)

# Comprehensive Report Export
comprehensive_report_output_path = os.path.join(PERFORMANCE_OUTPUT_DIRECTORY, "DT_Classification_Report.txt")
with open(comprehensive_report_output_path, "w") as report_file_handle:
    report_file_handle.write("="*60 + "\n")
    report_file_handle.write("Decision Tree Classification Comprehensive Report\n")
    report_file_handle.write("="*60 + "\n\n")
    report_file_handle.write("Performance Metrics Summary:\n")
    report_file_handle.write(performance_results_table.to_string(index=False))
    report_file_handle.write("\n\n" + "="*60 + "\n")
    report_file_handle.write("Detailed Metrics Breakdown:\n")
    report_file_handle.write("="*60 + "\n")
    report_file_handle.write(f"Macro Precision: {macro_precision_avg:.4f} | Macro Recall: {macro_recall_avg:.4f} | Macro F1: {macro_f1_avg:.4f}\n")
    report_file_handle.write(f"Micro Precision: {micro_precision_avg:.4f} | Micro Recall: {micro_recall_avg:.4f} | Micro F1: {micro_f1_avg:.4f}\n")
    report_file_handle.write(f"Weighted Precision: {weighted_precision_avg:.4f} | Weighted Recall: {weighted_recall_avg:.4f} | Weighted F1: {weighted_f1_avg:.4f}\n")
    report_file_handle.write("\n" + "="*60 + "\n")
    report_file_handle.write("Sklearn Classification Report:\n")
    report_file_handle.write("="*60 + "\n")
    report_file_handle.write(detailed_classification_report)

print(f"\nComprehensive report exported: {comprehensive_report_output_path}")


