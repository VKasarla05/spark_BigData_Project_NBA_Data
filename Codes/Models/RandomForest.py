#=========================================================
# Random Forest Model
# =========================================================
# Import Libraries
# =========================================================
import os
import time as execution_timer
from pyspark.sql import SparkSession as ClusterCompute
from pyspark.sql import functions as dataframe_utilities
from pyspark.ml.feature import VectorAssembler as ColumnAggregator
from pyspark.ml.feature import StandardScaler as FeatureNormalizer
from pyspark.ml.classification import RandomForestClassifier as TreeBasedClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as CategoryMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator as TwoClassMetrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as data_table_library
import matplotlib.pyplot as chart_generator
import seaborn as stats_visualizer

# Data Path
PLAYER_STATS_FILE_PATTERN = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
OUTPUT_ARTIFACTS_FOLDER = "/content/"

TRAIN_SPLIT_PERCENTAGE = 0.8
TEST_SPLIT_PERCENTAGE = 0.2
RANDOM_NUMBER_SEED = 42
NUMBER_OF_DECISION_TREES = 200
MAXIMUM_TREE_DEPTH = 12
FEATURE_SELECTION_METHOD = "auto"

# =========================================================
# Initializing Distributed Processing Engine
# =========================================================
distributed_compute_engine = ClusterCompute.builder.appName("RF_Basketball_Classifier").getOrCreate()
print("Distributed processing engine initialized")

# =========================================================
# Load Player Performance Dataset
# =========================================================
basketball_metrics_dataframe = distributed_compute_engine.read.csv(
    PLAYER_STATS_FILE_PATTERN,
    header=True,
    inferSchema=True
)
number_of_players = basketball_metrics_dataframe.count()
number_of_metrics = len(basketball_metrics_dataframe.columns)
print(f"Data loaded successfully: {number_of_players} player records, {number_of_metrics} statistical measures")

# =========================================================
# Output Path
# =========================================================
os.makedirs(OUTPUT_ARTIFACTS_FOLDER, exist_ok=True)
print(f"Output location configured: {OUTPUT_ARTIFACTS_FOLDER}")

# =========================================================
# Identify Target Performance Metric
# =========================================================
efficiency_rating_columns = [
    col_identifier for col_identifier in basketball_metrics_dataframe.columns
    if "PER" in col_identifier.upper()
]
if not efficiency_rating_columns:
    raise RuntimeError("Player efficiency rating column not found")

target_efficiency_column = efficiency_rating_columns[0]
print(f"Target performance metric identified: {target_efficiency_column}")

# =========================================================
# Extract Numerical Feature Set
# =========================================================
numeric_predictor_columns = []
for column_identifier, column_datatype in basketball_metrics_dataframe.dtypes:
    if column_datatype in ("int", "double", "float", "bigint"):
        if column_identifier != target_efficiency_column:
            numeric_predictor_columns.append(column_identifier)

print(f"Number of numerical predictors: {len(numeric_predictor_columns)}")

# =========================================================
# Calculate Performance Threshold
# =========================================================
average_efficiency_score = basketball_metrics_dataframe.select(
    dataframe_utilities.mean(dataframe_utilities.col(target_efficiency_column))
).first()[0]
print(f"Performance threshold (average): {average_efficiency_score:.2f}")

# =========================================================
# Generate Binary Classification Labels
# =========================================================
dataset_with_binary_labels = basketball_metrics_dataframe.withColumn(
    "label",
    dataframe_utilities.when(
        dataframe_utilities.col(target_efficiency_column) >= average_efficiency_score, 1
    ).otherwise(0)
)

# =========================================================
# Prepare Feature Vectors
# =========================================================
dataset_without_missing = dataset_with_binary_labels.na.drop(subset=[target_efficiency_column])

column_aggregation_tool = ColumnAggregator(
    inputCols=numeric_predictor_columns,
    outputCol="raw_features"
)
dataset_with_feature_vectors = column_aggregation_tool.transform(dataset_without_missing)

# =========================================================
# Apply Feature Standardization
# =========================================================
feature_normalization_tool = FeatureNormalizer( inputCol="raw_features",outputCol="features",withMean=True,withStd=True)
normalization_model_fitted = feature_normalization_tool.fit(dataset_with_feature_vectors)
dataset_with_normalized_features = normalization_model_fitted.transform(dataset_with_feature_vectors)
final_modeling_dataset = dataset_with_normalized_features.select("features", "label")

# =========================================================
# Split into Training and Testing Sets
# =========================================================
training_dataset, testing_dataset = final_modeling_dataset.randomSplit( [TRAIN_SPLIT_PERCENTAGE, TEST_SPLIT_PERCENTAGE],seed=RANDOM_NUMBER_SEED)
num_training_samples = training_dataset.count()
num_testing_samples = testing_dataset.count()
print(f"Data partitioned: {num_training_samples} training samples, {num_testing_samples} test samples")

# =========================================================
# Visualize Performance Distribution
# =========================================================
efficiency_distribution_data = basketball_metrics_dataframe.select(target_efficiency_column).toPandas()

chart_generator.figure(figsize=(9, 5))
stats_visualizer.histplot(
    efficiency_distribution_data[target_efficiency_column],
    bins=35,
    kde=True,
    edgecolor='black',
    color='teal'
)
chart_generator.axvline(
    average_efficiency_score,
    linestyle='--',
    linewidth=2,
    color='red',
    label=f'Threshold = {average_efficiency_score:.2f}'
)
chart_generator.title("Player Efficiency Rating (PER) Distribution", fontsize=14, weight='bold')
chart_generator.xlabel("PER Value")
chart_generator.ylabel("Player Count")
chart_generator.legend()
chart_generator.tight_layout()
distribution_chart_path = os.path.join(OUTPUT_ARTIFACTS_FOLDER, "PER_Distribution_RF.png")
chart_generator.savefig(distribution_chart_path)
chart_generator.close()
print("Distribution chart created")

# =========================================================
# Configure Tree Ensemble Model
# =========================================================
tree_ensemble_model = TreeBasedClassifier(
    featuresCol="features",
    labelCol="label",
    probabilityCol="probability",
    rawPredictionCol="rawPrediction",
    predictionCol="prediction",
    numTrees=NUMBER_OF_DECISION_TREES,
    maxDepth=MAXIMUM_TREE_DEPTH,
    seed=RANDOM_NUMBER_SEED,
    featureSubsetStrategy=FEATURE_SELECTION_METHOD
)

# =========================================================
# Execute Model Training
# =========================================================
model_training_start = execution_timer.time()
trained_tree_ensemble = tree_ensemble_model.fit(training_dataset)
model_training_duration = execution_timer.time() - model_training_start
print(f"Model training completed in {model_training_duration:.2f} seconds")

# =========================================================
# Generate Test Predictions
# =========================================================
test_set_predictions = trained_tree_ensemble.transform(testing_dataset).cache()

# =========================================================
# Calculate Performance Metrics
# =========================================================
accuracy_metric_calculator = CategoryMetrics(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
precision_metric_calculator = CategoryMetrics(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
recall_metric_calculator = CategoryMetrics(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)
f1_metric_calculator = CategoryMetrics(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
auc_metric_calculator = TwoClassMetrics(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

calculated_accuracy = accuracy_metric_calculator.evaluate(test_set_predictions)
calculated_precision = precision_metric_calculator.evaluate(test_set_predictions)
calculated_recall = recall_metric_calculator.evaluate(test_set_predictions)
calculated_f1_score = f1_metric_calculator.evaluate(test_set_predictions)
calculated_auc_score = auc_metric_calculator.evaluate(test_set_predictions)

# =========================================================
# Display and Save Results
# =========================================================
print("\nModel Performance Summary:")
performance_metrics_table = data_table_library.DataFrame({
    "Model": ["Random Forest Classifier"],
    "Accuracy": [calculated_accuracy],
    "Precision": [calculated_precision],
    "Recall": [calculated_recall],
    "F1": [calculated_f1_score],
    "AUC": [calculated_auc_score],
    "Training_Time(s)": [model_training_duration]
})
print(performance_metrics_table.to_string(index=False, justify='center'))

metrics_output_file = os.path.join(OUTPUT_ARTIFACTS_FOLDER, "RF_Classification_Results.csv")
performance_metrics_table.to_csv(metrics_output_file, index=False)
print(f"Metrics saved: {metrics_output_file}")

# =========================================================
# Create Confusion Matrix Visualization
# =========================================================
true_vs_predicted_labels = test_set_predictions.select("label", "prediction").toPandas()
classification_error_matrix = confusion_matrix(
    true_vs_predicted_labels["label"],
    true_vs_predicted_labels["prediction"]
)

matrix_visualization_object = ConfusionMatrixDisplay(
    confusion_matrix=classification_error_matrix,
    display_labels=["Below Avg", "Above/Equal Avg"]
)
matrix_visualization_object.plot(cmap="Greens")
chart_generator.title("Random Forest Classification Confusion Matrix", fontsize=14, weight='bold')
chart_generator.tight_layout()
confusion_matrix_file = os.path.join(OUTPUT_ARTIFACTS_FOLDER, "Confusion_Matrix_RF.png")
chart_generator.savefig(confusion_matrix_file)
chart_generator.close()
print("Confusion matrix chart saved")

# =========================================================
# Create ROC Curve Visualization
# =========================================================
probability_predictions_table = test_set_predictions.select("label", "probability").toPandas()
class_one_probability_scores = probability_predictions_table["probability"].apply(lambda v: float(v[1]))

false_positive_rate_values, true_positive_rate_values, _ = roc_curve(
    probability_predictions_table["label"],
    class_one_probability_scores
)
roc_curve_auc_score = roc_auc_score(
    probability_predictions_table["label"],
    class_one_probability_scores
)

chart_generator.figure(figsize=(7, 5))
chart_generator.plot(
    false_positive_rate_values,
    true_positive_rate_values,
    lw=2,
    color='blue',
    label=f"AUC = {roc_curve_auc_score:.3f}"
)
chart_generator.plot([0, 1], [0, 1], linestyle="--", color='gray')
chart_generator.xlabel("False Positive Rate")
chart_generator.ylabel("True Positive Rate")
chart_generator.title("ROC Curve — Random Forest")
chart_generator.legend(loc="lower right")
chart_generator.tight_layout()
roc_curve_file = os.path.join(OUTPUT_ARTIFACTS_FOLDER, "ROC_Curve_RF.png")
chart_generator.savefig(roc_curve_file)
chart_generator.close()
print("ROC curve chart saved")

# =========================================================
# Create Precision-Recall Curve Visualization
# =========================================================
precision_score_values, recall_score_values, _ = precision_recall_curve(
    probability_predictions_table["label"],
    class_one_probability_scores
)
precision_recall_average_precision = average_precision_score(
    probability_predictions_table["label"],
    class_one_probability_scores
)

chart_generator.figure(figsize=(7, 5))
chart_generator.plot(
    recall_score_values,
    precision_score_values,
    lw=2,
    color='darkorange',
    label=f"AP = {precision_recall_average_precision:.3f}"
)
chart_generator.xlabel("Recall")
chart_generator.ylabel("Precision")
chart_generator.title("Precision–Recall Curve — Random Forest")
chart_generator.legend(loc="lower left")
chart_generator.tight_layout()
precision_recall_curve_file = os.path.join(OUTPUT_ARTIFACTS_FOLDER, "Precision_Recall_Curve_RF.png")
chart_generator.savefig(precision_recall_curve_file)
chart_generator.close()
print("Precision-recall curve chart saved")


