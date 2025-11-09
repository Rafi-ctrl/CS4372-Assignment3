# CNN Transfer Learning (Colab) — Final Submission Guide

This README explains exactly how to **run, evaluate, and export** everything required for the assignment in **Google Colab** using **MobileNetV2** transfer learning.

---

## ✅ Deliverables Checklist
- [x] Training curves (**accuracy & loss** vs epochs)
- [x] Transfer learning + fine-tuning
- [x] **25 test samples** with true/pred labels (image grid + CSV)
- [x] **Confusion matrix** (normalized)
- [x] **Classification report** (saved to CSV)
- [x] **Experiments table** (iterations + params + results, saved to CSV)
- [x] **Saved model** (`.keras`)
- [x] Short notes on data split and assumptions

**Saved files:**
outputs_best_model.keras
outputs_plots_acc_loss.png
outputs_plots_loss.png
outputs_confusion_matrix_norm.png
outputs_sample25.png
outputs_test_samples_25.csv
outputs_classification_report.csv
experiments_log.csv


---

## 1) Colab Runtime & Setup
- **Runtime → Change runtime type → Hardware accelerator = GPU** (A100 or T4).

```python
# (Optional) Reproducibility
import os, random, numpy as np, tensorflow as tf
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
2) Data Options (pick one)
Option A — Kaggle (example: Stanford Dogs)
```
Upload kaggle.json to Colab (left Files panel).

Place credentials & set permissions:
```
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Download & unzip dataset (example; replace if needed):
```
kaggle datasets download -d jessicali9530/stanford-dogs-dataset -p /content/data
unzip -q /content/data/stanford-dogs-dataset.zip -d /content/data/raw
```
Option B — Google Drive

Place your dataset with class subfolders (e.g., dogs/pug, dogs/husky, …):
```
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/your_dataset_root"  # <-- change me
```

3) Build Datasets (train / validation-as-test)

If the dataset has no official test split, it’s acceptable to treat validation as a held-out test.
```
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2
DATA_DIR = "/content/data/raw/Images"   # <-- change to your folder with class subfolders

train_ds_p = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="training", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical"
)
val_ds_p = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="validation", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False
)
class_names = train_ds_p.class_names

aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
def preprocess(x, y):
    x = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(x, tf.float32))
    return x, y

AUTO = tf.data.AUTOTUNE
train_ds_p = train_ds_p.map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=AUTO)\
                       .map(preprocess, num_parallel_calls=AUTO).prefetch(AUTO)
val_ds_p   = val_ds_p.map(preprocess, num_parallel_calls=AUTO).prefetch(AUTO)
```
4) Model + Two-Phase Training (MobileNetV2)
```
import tensorflow as tf
from tensorflow.keras import layers

BASE_LR = 5e-4
EPOCHS_FROZEN   = 5
EPOCHS_FINETUNE = 5
UNFREEZE_FROM   = 100  # try 80 / 100 / 120 across iterations

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
)
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

# Phase 1: feature extractor
base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(BASE_LR),
              loss="categorical_crossentropy", metrics=["accuracy"])
hist1 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=EPOCHS_FROZEN)

# Phase 2: fine-tune
base_model.trainable = True
for layer in base_model.layers[:UNFREEZE_FROM]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(BASE_LR/10),
              loss="categorical_crossentropy", metrics=["accuracy"])
hist2 = model.fit(train_ds_p, validation_data=val_ds_p, epochs=EPOCHS_FINETUNE)

# Merge histories for plotting
history = {k: hist1.history.get(k, []) + hist2.history.get(k, [])
           for k in set(hist1.history) | set(hist2.history)}


Training curves

import matplotlib.pyplot as plt
epochs = range(1, len(history["loss"])+1)

plt.figure(); plt.plot(epochs, history["accuracy"]); plt.plot(epochs, history["val_accuracy"])
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(["train","val"]); plt.title("Accuracy vs Epochs")
plt.savefig("outputs_plots_acc_loss.png", bbox_inches="tight"); plt.show()

plt.figure(); plt.plot(epochs, history["loss"]); plt.plot(epochs, history["val_loss"])
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(["train","val"]); plt.title("Loss vs Epochs")
plt.savefig("outputs_plots_loss.png", bbox_inches="tight"); plt.show()

```
Save model
```
model.save("outputs_best_model.keras")
```

5) Evaluate (use validation as test)
```
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

test_ds = val_ds_p

# y_true
y_true = []
for _, y in test_ds:
    y_true.extend(np.argmax(y.numpy(), axis=1))
y_true = np.array(y_true)

probs = model.predict(test_ds, verbose=0)
y_pred = probs.argmax(axis=1)

present = np.unique(y_true)
present_names = [class_names[i] for i in present]

cm = confusion_matrix(y_true, y_pred, labels=present)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12,10))
sns.heatmap(cm_norm, xticklabels=present_names, yticklabels=present_names,
            cmap="Blues", square=True, cbar=False)
plt.title("Confusion Matrix (validation-as-test) — normalized")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig("outputs_confusion_matrix_norm.png", bbox_inches="tight"); plt.show()

print(classification_report(
    y_true, y_pred, labels=present, target_names=present_names, digits=3, zero_division=0
))

# Save classification report as CSV
from sklearn.metrics import classification_report as _cr
import pandas as pd
rep = _cr(y_true, y_pred, labels=present, target_names=present_names, digits=3, output_dict=True, zero_division=0)
pd.DataFrame(rep).to_csv("outputs_classification_report.csv")


25-sample grid + CSV

# Single-image pipeline to get file paths
single = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="validation", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=1, label_mode="categorical", shuffle=False
)
paths = single.file_paths

import pandas as pd, math
take_n = 25
rows = []
for i, (img, label) in enumerate(single.take(take_n)):
    prob = model.predict(img, verbose=0)[0]
    true_idx = int(np.argmax(label.numpy()[0])); pred_idx = int(np.argmax(prob))
    rows.append({"data_path": paths[i],
                 "true_label": class_names[true_idx],
                 "pred_label": class_names[pred_idx]})
df = pd.DataFrame(rows)
df.to_csv("outputs_test_samples_25.csv", index=False)

# Grid
plt.figure(figsize=(12,12))
cols = int(math.ceil(math.sqrt(take_n))); rws = int(math.ceil(take_n/cols))
single2 = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="validation", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=1, label_mode="categorical", shuffle=False
)
for i, (img, _) in enumerate(single2.take(take_n)):
    plt.subplot(rws, cols, i+1); plt.imshow(tf.cast(img[0], tf.uint8)); plt.axis("off")
    plt.title(f"T:{df.true_label[i]}\nP:{df.pred_label[i]}", fontsize=8)
plt.tight_layout(); plt.savefig("outputs_sample25.png", bbox_inches="tight"); plt.show()
```

6) Experiments Table (iterations + params + results)
```
import pandas as pd, time
row = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "base_lr": BASE_LR,
    "epochs_frozen": EPOCHS_FROZEN, "epochs_finetune": EPOCHS_FINETUNE,
    "unfreeze_from": UNFREEZE_FROM,
    "train_acc_final": history["accuracy"][-1],
    "val_acc_final": history["val_accuracy"][-1],
}
LOG_PATH = "experiments_log.csv"
try:
    old = pd.read_csv(LOG_PATH)
    log = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
except FileNotFoundError:
    log = pd.DataFrame([row])
log.to_csv(LOG_PATH, index=False)
log


Suggested iterations to try (and log):

UNFREEZE_FROM = 80 / 100 / 120

BASE_LR = 5e-4 / 1e-4

BATCH_SIZE = 32 / 64 (if VRAM allows)
```
7) Notes (add a short cell at top)
```
GPU: Colab A100/T4 used.

Split: validation_split={VAL_SPLIT}; validation treated as held-out test.

Classes: len(class_names) discovered; IMG_SIZE=224.

Two-phase training: frozen → fine-tune from UNFREEZE_FROM.
```
8) Troubleshooting

5D input error (None, None, 224, 224, 3) → don’t batch twice; rebuild datasets once per run.

OOM → reduce BATCH_SIZE or image size; keep augmentation light.

Zero-division warnings → pass labels=present & zero_division=0 (as shown).


---

# 2) Quick `.gitignore` (optional but recommended)

```gitignore
# Python / Colab
__pycache__/
.ipynb_checkpoints/

# Large artifacts (don’t version big binaries)
*.h5
*.keras
*.ckpt
*.tflite
outputs_best_model.keras
*.zip

# Data
data/
content/
```
