from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName('signLanguageCNN').getOrCreate()

#Loading the train and test path
train_path = '/content/sign_mnist_train.csv'
test_path = '/content/sign_mnist_test.csv'

# Converting into dataframes
train_df = spark.read.csv(train_path, header=True, inferSchema=True)
test_df = spark.read.csv(test_path, header=True, inferSchema=True)

# Vector conversion
pixel_columns = [f'pixel{i}' for i in range(1, 785)]

assembler_CNN = VectorAssembler(
    inputCols = pixel_columns,
    outputCol = 'features'
)

#Transforming the dataframes
train_df_tra = assembler_CNN.transform(train_df).select('features', 'label')
test_df_tra = assembler_CNN.transform(test_df).select('features', 'label')

def to_numpy(df):
  features = np.array(df.select('features').rdd.map(lambda x: x[0]).collect())
  label = np.array(df.select('label').rdd.map(lambda x: x[0]).collect())
  return features, label

X_train, y_train = to_numpy(train_df_tra)
X_test, y_test = to_numpy(test_df_tra)

y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)



#Reshape for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#Build CNN in tensorflow
model_CNN = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')

])

model_CNN.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model_CNN.summary()

# Train the CNN model
history = model_CNN.fit(
    X_train, y_train,
    epochs = 10,
    batch_size = 64,
    validation_split = 0.2)
test_loss, test_acc = model_CNN.evaluate(X_test, y_test)
print('Test Accuracy for Model CNN', test_acc)

# Checking for Accuracy training using CNN Model
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Accuracy for Each Epoch')
plt.legend()
plt.show()

#Checking for Loss training using CNN Model
plt.plot(history.history['loss'], label = 'Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Accuracy for Each Epoch')
plt.legend()
plt.show()

# Visualizing Confusion Matrix
y_pred = model_CNN.predict(X_test)
y_pred_classes = y_pred.argmax(axis = 1)

cm = confusion_matrix(y_test, y_pred_classes)

sns.heatmap(cm, annot=False, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predictions')
plt.ylabel('Actual')
plt.show()

