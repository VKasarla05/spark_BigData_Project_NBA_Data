# ----------- 1. Setup and Imports -----------
# Make sure you have the necessary packages installed in your environment
# If using a notebook, run this in a cell before running the rest of the code:
# !pip install pyspark tensorflow scikit-learn matplotlib

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession
import itertools

# ----------- 2. Start Spark -----------
spark = SparkSession.builder \
    .appName("Spark CNN Image Project") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark started:", spark.sparkContext.master)

# ----------- 3. Get List of Images -----------
data_path = "/home/sat3812/CNNProject/Dataset"  # Set this to your image folder

df = spark.read.format("binaryFile") \
    .option("recursiveFileLookup", "true") \
    .load(data_path)

def extract_paths(df):
    # Strips 'file:' from Spark paths, gets real file paths
    paths = []
    for row in df.select("path").collect():
        p = row.path
        if p.startswith("file:"):
            p = p.replace("file:", "", 1)
        paths.append(p)
    return paths

# Split: 70% train, 15% val, 15% test
train_df, val_df, test_df = df.randomSplit([0.7,0.15,0.15], seed=42)
train_files = np.array(extract_paths(train_df), dtype=str)
val_files   = np.array(extract_paths(val_df), dtype=str)
test_files  = np.array(extract_paths(test_df), dtype=str)

print('Train files:', len(train_files), 'Val files:', len(val_files), 'Test files:', len(test_files))

if len(train_files) == 0:
    print("No training files found. Check your path.")
    exit()

# ----------- 4. Setup Output Folder -----------
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Outputs will be saved in", OUTPUT_DIR)

# ----------- 5. Prepare Data for TensorFlow -----------
# All images resized to 48x48 grayscale for simplicity

IMG_SIZE = (48, 48)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32) / 255.0

def get_label(path):
    # Use filename for label (simple method)
    path = path.numpy().decode("utf-8")
    return 0 if "fracture" in path.lower() else 1

def load_pair(path):
    img = load_image(path)
    label = tf.py_function(get_label, [path], tf.int32)
    label.set_shape([])  # Make sure it's scalar
    return img, label

# Build TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices(train_files) \
    .shuffle(len(train_files)) \
    .map(load_pair, num_parallel_calls=AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(val_files) \
    .map(load_pair, num_parallel_calls=AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_files) \
    .map(load_pair, num_parallel_calls=AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(AUTOTUNE)

# ----------- 6. Visualize Sample Images -----------
plt.figure(figsize=(10,6))
for i, (img, label) in enumerate(train_ds.take(6)):
    plt.subplot(2,3,i+1)
    plt.imshow(img[0].numpy().squeeze(), cmap="gray")
    plt.title("fracture" if label[0].numpy() == 0 else "normal")
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))

# ----------- 7. Build CNN Model -----------
inputs = tf.keras.Input(shape=(48,48,1))
x = tf.keras.layers.Conv2D(32,3,activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64,3,activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128,3,activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128,activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(2,activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda line: f.write(line + "\n"))

# ----------- 8. Train Model -----------
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop])

# ----------- 9. Show Training Curves -----------
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))

# ----------- 10. Test Evaluation -----------
test_loss, test_acc = model.evaluate(test_ds)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)
with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc}\nTest Loss: {test_loss}\n")

# ----------- 11. Report and Confusion Matrix -----------
y_true, y_pred = [], []
for imgs, labels in test_ds:
    preds = model.predict(imgs)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
report = classification_report(y_true, y_pred, target_names=["fracture", "normal"])
print(report)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
classes = ["fracture", "normal"]
plt.xticks(range(2), classes)
plt.yticks(range(2), classes)
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > cm.max()/2 else "black")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# ----------- 12. Grad-CAM Visual Explanation -----------
last_conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1].name
def make_gradcam(img_array):
    grad_model = tf.keras.Model(inputs=model.input,
                               outputs=[model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_idx = tf.argmax(preds[0])
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_out[0], axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / tf.reduce_max(cam)
    return cam.numpy()

for imgs, labels in test_ds.take(1):
    img = imgs[0:1]
    heat = make_gradcam(img)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0].numpy().squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img[0].numpy().squeeze(), cmap="gray")
    plt.imshow(heat, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gradcam.png"))
    break

# ----------- 13. Stop Spark -----------
spark.stop()
print("Spark stopped.")
