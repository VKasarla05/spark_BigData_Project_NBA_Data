# ================== SINGLE-CELL PySpark + CNN FOR INTEL 6-CLASS DATA ==================

import os, time, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from sklearn.metrics import classification_report, confusion_matrix
import itertools

# ---------------------- 1. Start Timers ----------------------
script_start = time.time()

# ---------------------- 2. Start Spark -----------------------
spark = SparkSession.builder \
    .appName("Spark_CNN_IntelSceneClassification") \
    .master("local[*]") \
    .config("spark.driver.memory","4g") \
    .getOrCreate()

print("Spark Started:", spark.sparkContext.master)

# ---------------------- 3. Load Intel Dataset ----------------------
# Path example: /home/youruser/IntelDataset/seg_train/
data_path = "/home/sat3812/IntelDataset/seg_train"  # <<< CHANGE THIS TO YOUR PATH

df = spark.read.format("binaryFile") \
    .option("recursiveFileLookup", "true") \
    .load(data_path)

# Extract real paths
def extract_paths(df):
    paths = []
    for row in df.select("path").collect():
        p = row.path.replace("file:", "") if row.path.startswith("file:") else row.path
        paths.append(p)
    return paths

# Create train/val/test via Spark split
train_df, val_df, test_df = df.randomSplit([0.70, 0.15, 0.15], seed=42)

train_files = np.array(extract_paths(train_df), dtype=str)
val_files   = np.array(extract_paths(val_df), dtype=str)
test_files  = np.array(extract_paths(test_df), dtype=str)

print("Train:", len(train_files), "Val:", len(val_files), "Test:", len(test_files))

# ---------------------- 4. Output Folder ----------------------
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_intel")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Outputs will be saved in:", OUTPUT_DIR)

# ---------------------- 5. TF Dataset Pipeline ----------------------
IMG_SIZE = (128, 128)
AUTOTUNE = tf.data.AUTOTUNE
BATCH = 32

# Map folder names to integer labels
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32) / 255.0

def get_label(path):
    p = path.numpy().decode("utf-8").lower()
    for class_name in CLASS_NAMES:
        if class_name in p:
            return CLASS_TO_ID[class_name]
    return -1

def load_pair(path):
    img = load_image(path)
    lbl = tf.py_function(get_label, [path], tf.int32)
    lbl.set_shape([])
    return img, lbl

train_ds = tf.data.Dataset.from_tensor_slices(train_files).shuffle(10000) \
    .map(load_pair, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(val_files) \
    .map(load_pair, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_files) \
    .map(load_pair, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)

# ---------------------- 6. Sample Images ----------------------
plt.figure(figsize=(10,6))
for i, (img, lbl) in enumerate(train_ds.take(6)):
    plt.subplot(2,3,i+1)
    plt.imshow(img[0].numpy())
    plt.title(CLASS_NAMES[lbl[0].numpy()])
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))

# ---------------------- 7. Build CNN ----------------------
num_classes = len(CLASS_NAMES)

inputs = tf.keras.Input(shape=(128,128,3))
x = tf.keras.layers.Conv2D(32,3,activation="relu",padding="same")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64,3,activation="relu",padding="same")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128,3,activation="relu",padding="same")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256,activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(num_classes,activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# ---------------------- 8. Train Model ----------------------
print("\n=== TRAINING STARTED ===")
train_start = time.time()

early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop])

training_time = time.time() - train_start
with open(os.path.join(OUTPUT_DIR, "training_time.txt"), "w") as f:
    f.write(str(training_time))

# ---------------------- 9. Training Plots ----------------------
plt.figure(figsize=(14,5))
plt.subplot(1,2,1); plt.plot(history.history["accuracy"]); plt.plot(history.history["val_accuracy"]); plt.title("Accuracy")
plt.subplot(1,2,2); plt.plot(history.history["loss"]); plt.plot(history.history["val_loss"]); plt.title("Loss")
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))

# ---------------------- 10. Test Evaluation ----------------------
test_loss, test_acc = model.evaluate(test_ds)
with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
    f.write(f"Accuracy: {test_acc}\nLoss: {test_loss}")

# ---------------------- 11. Classification Report/Confusion Matrix ----------------------
y_true, y_pred = [], []
for imgs, lbls in test_ds:
    preds = model.predict(imgs)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(lbls.numpy())

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7,6))
plt.imshow(cm, cmap="Blues"); plt.colorbar(); plt.title("Confusion Matrix")
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i,j], ha="center", color="white" if cm[i,j]>cm.max()/2 else "black")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# ---------------------- 12. Stop Spark ----------------------
spark.stop()

print(f"\nTotal Runtime: {time.time() - script_start:.2f} sec")

# ================== END OF SCRIPT ==================
