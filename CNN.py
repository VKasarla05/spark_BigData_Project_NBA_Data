# ------------ 0. IMPORTS -----------
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report, confusion_matrix
from pyspark.sql import SparkSession

# ------------ 1. SPARK SESSION -----------
# If you have a Spark cluster master URL, set it here, e.g. "spark://hadoop1:7077"
# Otherwise, use local[*] to use all cores on this VM.
spark = SparkSession.builder \
    .appName("Spark CNN Image Project") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark session created with master:", spark.sparkContext.master)

# ------------ 2. LOAD IMAGE PATHS WITH SPARK -----------
# CHANGE THIS TO YOUR REAL PATH ON THE VM
data_path = "/home/your_username/datasets/bone_fracture"  # TODO: update this

df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .option("recursiveFileLookup", "true") \
    .load(data_path)

print("Total images:", df.count())
# ------------ 3. EXTRACT FILE PATHS FROM SPARK DF -----------
def extract_paths(df):
    return [row.path for row in df.select("path").collect()]

train_df, val_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=42)

train_files = np.array(extract_paths(train_df), dtype=str)
val_files   = np.array(extract_paths(val_df), dtype=str)
test_files  = np.array(extract_paths(test_df), dtype=str)

print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))

# ------------ 4. CREATE OUTPUT DIRECTORY -----------
os.makedirs("outputs", exist_ok=True)

# ------------ 5. TF DATASET LOADING -----------
IMG_SIZE = (48, 48)

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)   # supports PNG/JPG
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def get_label(path):
    path = path.numpy().decode("utf-8")
    return 0 if "fracture" in path.lower() else 1

def load_pair(path):
    img = load_image(path)
    label = tf.py_function(get_label, [path], tf.int32)
    label.set_shape([])  # Important for shape consistency
    return img, label

BATCH_SIZE = 32

train_ds = tf.data.Dataset.from_tensor_slices(train_files) \
    .shuffle(len(train_files)) \
    .map(load_pair, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(val_files) \
    .map(load_pair, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_files) \
    .map(load_pair, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)


# ------------ 6. SHOW SAMPLE IMAGES -----------
import matplotlib
matplotlib.use("Agg")

plt.figure(figsize=(10, 6))
for i, (img, label) in enumerate(train_ds.take(6)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img[0].numpy().squeeze(), cmap="gray")
    plt.title("fracture" if label[0].numpy() == 0 else "normal")
    plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_images.png")


# ------------ 7. BUILD CNN MODEL -----------
inputs = tf.keras.Input(shape=(48, 48, 1))

x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)

outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Save model summary
with open("outputs/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print("Model summary saved to outputs/model_summary.txt")


# ------------ 8. TRAIN MODEL -----------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop]
)

# ------------ 9. TRAINING CURVES -----------
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig("outputs/training_curves.png")


# ------------ 10. TEST EVALUATION -----------
test_loss, test_acc = model.evaluate(test_ds)
print("TEST ACCURACY:", test_acc)
print("TEST LOSS:", test_loss)

with open("outputs/test_metrics.txt", "w") as f:
    f.write(f"Test Accuracy: {test_acc}\nTest Loss: {test_loss}\n")


# ------------ 11. CLASSIFICATION REPORT -----------
y_true, y_pred = [], []

for imgs, labels in test_ds:
    preds = model.predict(imgs)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Save report
report = classification_report(y_true, y_pred, target_names=["fracture", "normal"])
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)

print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

classes = ["fracture", "normal"]
plt.xticks(range(2), classes)
plt.yticks(range(2), classes)

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")


# ------------ 12. GRAD-CAM -----------
last_conv = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1].name

def make_gradcam(img_array):
    grad_model = tf.keras.Model(model.input, outputs=[
        model.get_layer(last_conv).output,
        model.output
    ])

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

# Make Grad-CAM for 1 example
for imgs, labels in test_ds.take(1):
    img = imgs[0:1]
    heat = make_gradcam(img)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0].numpy().squeeze(), cmap="gray")
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(img[0].numpy().squeeze(), cmap="gray")
    plt.imshow(heat, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM")

    plt.tight_layout()
    plt.savefig("outputs/gradcam.png")
    break

spark.stop()

