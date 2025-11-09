
# MLP MODEL

# Importing Libraries
import os
import time as execution_chronometer
from pyspark.sql import SparkSession as AnalyticsComputeEngine
from pyspark.sql import functions as neural_operations
from pyspark.ml.feature import VectorAssembler as FeatureVectorizer
from pyspark.ml.feature import StandardScaler as ZScoreNormalizer
from pyspark.ml.classification import MultilayerPerceptronClassifier as NeuralNetworkClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as PerformanceMetricsEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator as BinaryPerformanceEvaluator
from sklearn.metrics import confusion_matrix as calculate_error_matrix
from sklearn.metrics import ConfusionMatrixDisplay as ErrorMatrixRenderer
import pandas as dataframe_handler
import matplotlib.pyplot as visualization_platform
import seaborn as statistical_visualizer


INPUT_DATASET_LOCATION = "/content/discretized_nba_stats/discretized_nba_stats/part-*.csv"
OUTPUT_ARTIFACTS_LOCATION = "/content/"
RANDOM_STATE_SEED = 42
NEURAL_NETWORK_LAYERS = [20, 10]
MAXIMUM_TRAINING_ITERATIONS = 100


# Spark Initialization
analytics_compute_engine = AnalyticsComputeEngine.builder.appName("NBA_MLP_Classification").getOrCreate()
# Dataset
sports_performance_dataset = analytics_compute_engine.read.csv(
    INPUT_DATASET_LOCATION,
    header=True,
    inferSchema=True
)
dataset_observation_count = sports_performance_dataset.count()
dataset_dimension_count = len(sports_performance_dataset.columns)
print(f"Athletic dataset acquisition successful: {dataset_observation_count} observations, {dataset_dimension_count} dimensions")

# Output location
os.makedirs(OUTPUT_ARTIFACTS_LOCATION, exist_ok=True)
print(f"Output artifacts repository configured: {OUTPUT_ARTIFACTS_LOCATION}")

# Target Performance
performance_target_columns = [
    column_identifier for column_identifier in sports_performance_dataset.columns
    if "PER" in column_identifier.upper()
]
identified_performance_target = performance_target_columns[0]
print(f"Performance target metric identified: {identified_performance_target}")

# Numerical Feature Attribute Extraction
extracted_numerical_features = [
    column_name for column_name, data_type in sports_performance_dataset.dtypes
    if data_type in ("int", "double", "float", "bigint") 
    and column_name != identified_performance_target
]
print(f"Numerical feature attributes extracted: {len(extracted_numerical_features)}")
# Feature Extraction
feature_vectorizer_transformer = FeatureVectorizer(
    inputCols=extracted_numerical_features,
    outputCol="raw_features"
)
aggregated_feature_dataset = feature_vectorizer_transformer.transform(
    sports_performance_dataset.na.drop(subset=[identified_performance_target])
)

# Feature Value Normalization via Z-Score Transformation
zscore_normalizer = ZScoreNormalizer(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)
normalization_model_instance = zscore_normalizer.fit(aggregated_feature_dataset)
normalized_feature_dataset = normalization_model_instance.transform(aggregated_feature_dataset)

# Performance Threshold Computation
computed_performance_threshold = sports_performance_dataset.select(
    neural_operations.mean(neural_operations.col(identified_performance_target))
).collect()[0][0]
print(f"Performance classification threshold (mean): {computed_performance_threshold:.2f}")

# Binary Classification Label Generation
dataset_with_classification_labels = sports_performance_dataset.withColumn(
    "label",
    neural_operations.when(
        neural_operations.col(identified_performance_target) >= computed_performance_threshold,
        1
    ).otherwise(0)
)

# Feature Re-aggregation for Classification Task
classification_feature_dataset = feature_vectorizer_transformer.transform(
    dataset_with_classification_labels.na.drop(subset=[identified_performance_target])
)
normalized_classification_dataset = normalization_model_instance.transform(
    classification_feature_dataset
).select("features", "label")

# Training-Testing Dataset Partitioning
model_training_subset, model_testing_subset = normalized_classification_dataset.randomSplit(
    [0.8,0.2],
    seed=RANDOM_STATE_SEED
)
training_subset_size = model_training_subset.count()
testing_subset_size = model_testing_subset.count()
print(f"Data partitioning completed: {training_subset_size} training observations, {testing_subset_size} testing observations")

# Performance Metric
performance_metric_distribution = sports_performance_dataset.select(identified_performance_target).toPandas()

visualization_platform.figure(figsize=(9, 5))
statistical_visualizer.histplot(
    performance_metric_distribution[identified_performance_target],
    bins=35,
    kde=True,
    color='darkorange',
    edgecolor='black'
)
visualization_platform.axvline(
    computed_performance_threshold,
    color='navy',
    linestyle='--',
    linewidth=2,
    label=f'Threshold = {computed_performance_threshold:.2f}'
)
visualization_platform.title("Player Efficiency Rating (PER) Distribution Analysis", fontsize=14, weight='bold')
visualization_platform.xlabel("PER Value", fontsize=12)
visualization_platform.ylabel("Player Observation Count", fontsize=12)
visualization_platform.legend()
visualization_platform.tight_layout()

distribution_visualization_output = os.path.join(OUTPUT_ARTIFACTS_LOCATION, "PER_Distribution.png")
visualization_platform.savefig(distribution_visualization_output, dpi=300)
visualization_platform.close()
print(f"Performance distribution visualization saved: {distribution_visualization_output}")

# Neural Network Architecture
input_layer_dimension = len(extracted_numerical_features)
complete_network_architecture = [input_layer_dimension] + NEURAL_NETWORK_LAYERS + [2]
print(f"Neural network architecture configured: {complete_network_architecture}")

neural_network_classifier = NeuralNetworkClassifier(
    featuresCol="features",
    labelCol="label",
    layers=complete_network_architecture,
    maxIter=MAXIMUM_TRAINING_ITERATIONS,
    seed=RANDOM_STATE_SEED
)

# Neural Network Model Training Execution
print("Initiating neural network training process...")
training_execution_start_timestamp = execution_chronometer.time()
fitted_neural_network_model = neural_network_classifier.fit(model_training_subset)
generated_predictions = fitted_neural_network_model.transform(model_testing_subset)
total_training_elapsed_time = execution_chronometer.time() - training_execution_start_timestamp
print(f"Neural network training process completed: {total_training_elapsed_time:.2f} seconds")

# Model Performance Evaluation Metrics Computation
accuracy_performance_evaluator = PerformanceMetricsEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
precision_performance_evaluator = PerformanceMetricsEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
recall_performance_evaluator = PerformanceMetricsEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)
auc_performance_evaluator = BinaryPerformanceEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

computed_accuracy_metric = accuracy_performance_evaluator.evaluate(generated_predictions)
computed_precision_metric = precision_performance_evaluator.evaluate(generated_predictions)
computed_recall_metric = recall_performance_evaluator.evaluate(generated_predictions)
computed_auc_metric = auc_performance_evaluator.evaluate(generated_predictions)

# Performance Results
performance_results_compilation = dataframe_handler.DataFrame({
    "Model": ["Multilayer Perceptron Neural Classifier"],
    "Accuracy": [computed_accuracy_metric],
    "Precision": [computed_precision_metric],
    "Recall": [computed_recall_metric],
    "AUC": [computed_auc_metric],
    "Training_Time(s)": [total_training_elapsed_time]
})

print("\nPerformance Metrics Summary:")
print(performance_results_compilation.to_string(index=False))

# Error Matrix Generation and Visualization
predicted_labels_dataframe = generated_predictions.select("label", "prediction").toPandas()
computed_classification_error_matrix = calculate_error_matrix(
    predicted_labels_dataframe["label"],
    predicted_labels_dataframe["prediction"]
)

error_matrix_visualization = ErrorMatrixRenderer(
    confusion_matrix=computed_classification_error_matrix,
    display_labels=["Below Average Performance", "Above Average Performance"]
)
error_matrix_visualization.plot(cmap="Oranges")
visualization_platform.title("Multilayer Perceptron Classification — Error Matrix Visualization", fontsize=14, weight='bold')
visualization_platform.tight_layout()

error_matrix_output_location = os.path.join(OUTPUT_ARTIFACTS_LOCATION, "Confusion_Matrix.png")
visualization_platform.savefig(error_matrix_output_location, dpi=300)
visualization_platform.close()
print(f"Error matrix visualization exported: {error_matrix_output_location}")
