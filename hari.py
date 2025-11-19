# ------------------ Food Classification using Pyspark& CNN------------------

import os
import numpy as np
from PIL import Image
from pyspark.sql import SparkSession, Row
import tensorflow as tf
from tensorflow.keras import layers, models

# ------------------ 1. START SPARK ------------------
spark = SparkSession.builder \
    .appName("Food_3Class_CNN_Spark") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark Started:", spark.sparkContext.master)

# ------------------ 2. LOAD DATASET ------------------
DATA_ROOT = "/home/sat3812/FoodDataset"   
CLASSES = ["meal", "drink", "dessert"]

file_paths = []
for cls in CLASSES:
    folder = os.path.join(DATA_ROOT, cls)
    for f in os.listdir(folder):
        if f.endswith((".jpg", ".jpeg", ".png")):
            file_paths.append((os.path.join(folder, f), cls))

print("Total Images Found:", len(file_paths))

rows = [Row(path=p, label_str=l) for p, l in file_paths]
df_paths = spark.createDataFrame(rows)
df_paths.show(5, truncate=False)

# ------------------ 3. LABEL INDEXING ------------------
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="label_str", outputCol="label_idx")
df_indexed = Pipeline(stages=[indexer]).fit(df_paths).transform(df_paths)

# ------------------ 4. TRAIN/VAL/TEST SPLIT ------------------
train_df, val_df, test_df = df_indexed.randomSplit([0.7, 0.15, 0.15], seed=42)
print("Train:", train_df.count(), "Val:", val_df.count(), "Test:", test_df.count())

train_paths = [(r["path"], int(r["label_idx"])) for r in train_df.collect()]
val_paths   = [(r["path"], int(r["label_idx"])) for r in val_df.collect()]
test_paths  = [(r["path"], int(r["label_idx"])) for r in test_df.collect()]

# ------------------ 5. CONVERT IMAGES INTO NUMPY ARRAYS ------------------
IMG_SIZE = 128

def load_images(path_list):
    X_list, y_list = [], []
    for path, label in path_list:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img) / 255.0
            X_list.append(arr)
            y_list.append(label)
        except Exception as e:
            print("Error loading:", path, "->", e)
    return np.array(X_list), np.array(y_list)

X_train, y_train = load_images(train_paths)
X_val,   y_val   = load_images(val_paths)
X_test,  y_test  = load_images(test_paths)

print("Shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape,   y_val.shape)
print("Test: ", X_test.shape,  y_test.shape)

NUM_CLASSES = len(CLASSES)

# ------------------ 6. BUILD CNN MODEL ------------------
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_cnn()
model.summary()

# ------------------ 7. TRAIN MODEL ------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

# ------------------ 8. EVALUATE MODEL ------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nFINAL TEST ACCURACY:", test_acc)

# ------------------ 9. SAVE OUTPUTS ------------------
OUTPUT_DIR = "/home/sat3812/FoodOutputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save(os.path.join(OUTPUT_DIR, "food_cnn_model.h5"))

# Save training curves
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"))
plt.clf()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.clf()

# Save confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# Save classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred, target_names=CLASSES))

print("\nAll outputs saved to:", OUTPUT_DIR)

spark.stop()
