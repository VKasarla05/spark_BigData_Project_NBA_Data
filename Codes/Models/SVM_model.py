# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import RandomForestClassifier, LinearSVC
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

# Starting the spark session
spark = SparkSession.builder.appName('NBA_RF_Classification').getOrCreate()

# Loading the dataset
data_path = '/content/part-*.csv'
nba_data = spark.read.csv(data_path, header=True, inferSchema=True)
print(nba_data)

#Feature Preperation

#Giving the target data
target = [c for c in nba_data.columns if 'PER' in c.upper()][0]
print(target)

#Selecting numeric columns
numeric_columns = [c for c, t in nba_data.dtypes if t in ('int', 'double', 'float', 'bigint') and c != target]

# Drop rows with missing values
clean_nba_data = nba_data.na.drop(subset=[target])

# Combining columns into a single feature column
assemble_data = VectorAssembler(inputCols=numeric_columns, outputCol='feature_raw_columns')
assembled_data = assemble_data.transform(clean_nba_data)

#Standardizing the data
scaler = StandardScaler(inputCol='feature_raw_columns', outputCol='features', withMean=True, withStd=True)
scaled_nba_data = scaler.fit(assembled_data).transform(assembled_data)

#Binary Label Creation

# Computing average of target variable
nba_data.createOrReplaceTempView('nba')

labele_data = spark.sql(f"""
SELECT *,
        CASE WHEN {target} >= (SELECT AVG({target}) FROM nba) THEN 1 ELSE 0 END AS label
  FROM nba
""")

labele_data.select(target, 'label').show()

# Setting features for classification
data_with_features = assemble_data.transform(labele_data.na.drop(subset=[target]))

# Standardizing the column
classification_data_scaling = (
    scaler.fit(data_with_features)
          .transform(data_with_features)
          .select('features', 'label')
)

# Splitting the data
train_data, test_data = classification_data_scaling.randomSplit([0.8, 0.2], seed=45)
print(f'Train set: {train_data.count()} | Test set: {test_data.count()}')

PER_values = nba_data.select(target).toPandas()

#visualizing the data
plt.hist(PER_values[target], bins=30, color='skyblue', edgecolor='black')
plt.title('Target Variable Distribution')
plt.xlabel(target)
plt.ylabel('Frequency')
plt.show()

# Building the SVM model
SVM = LinearSVC(featuresCol='features', labelCol='label', maxIter=100, regParam=0.1)

SVM_model = SVM.fit(train_data)
SVM_predictions = SVM_model.transform(test_data)

#Evaluation Matrix
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='accuracy'
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol='label', predictionCol='prediction', metricName='f1'
)

#Compute Metrix
accuracy = accuracy_evaluator.evaluate(SVM_predictions)
f1_score = f1_evaluator.evaluate(SVM_predictions)

print('SVM classification Results')
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1_score}')

SVM_predictions.select('label', 'prediction', 'features').show(5, truncate=False)

#Printing first model coefficients and intercepts
print(f'Coefficients of svm', SVM_model.coefficients[:5])
print(f'Intercepts of svm', SVM_model.intercept)

# Predictions to pandas
prediction_SVM = SVM_predictions.select('label', 'prediction').toPandas()
cm = confusion_matrix(prediction_SVM['label'], prediction_SVM['prediction'])
print('Confusion Matrix', cm)

display_pred = ConfusionMatrixDisplay(cm, display_labels=['Low_PER', 'High_PER'])
display_pred.plot(cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.show()

SVM_model.save('clean_output/SVM_NBA_MODEL')
