# src/train_cnn_radioml.py - CNN for RadioML 2016.10A
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

print("=" * 70)
print("RADIOML 2016.10A - CNN CLASSIFICATION")
print("=" * 70)

 
# 1. LOAD DATA
 
print("\n[1] Loading dataset...")
X = np.load('data/radioml_X.npy')  # Shape: (220000, 2, 128)
Y = np.load('data/radioml_Y.npy')  # Shape: (220000,)
Z = np.load('data/radioml_SNR.npy')  # Shape: (220000,)

with open('data/modulations.json', 'r') as f:
    modulations = json.load(f)

print(f"    Samples: {len(X)}")
print(f"    IQ shape: {X.shape[1:]}")
print(f"    Modulations: {len(modulations)}")
print(f"    SNR range: {np.min(Z):.0f}dB to {np.max(Z):.0f}dB")

 
# 2. PREPARE DATA FOR CNN
 
print("\n[2] Preparing data for CNN...")

# CNN expects: (batch, height, width, channels)
# Our data: (samples, 2, 128) → reshape to (samples, 2, 128, 1)
X_cnn = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
print(f"    CNN input shape: {X_cnn.shape}")

# Encode labels to integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
num_classes = len(label_encoder.classes_)
print(f"    Classes: {num_classes}")

 
# 3. TRAIN/TEST SPLIT
 
print("\n[3] Splitting data...")
X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    X_cnn, Y_encoded, Z, test_size=0.2, random_state=42, stratify=Y_encoded
)
print(f"    Train: {len(X_train)} samples")
print(f"    Test: {len(X_test)} samples")

 
# 4. BUILD CNN MODEL
 
print("\n[4] Building CNN model...")

def build_cnn(input_shape=(2, 128, 1), num_classes=11):
    """Build CNN for RF signal classification"""
    
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (1, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (1, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (1, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (1, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = build_cnn()
model.summary()

 
# 5. COMPILE MODEL
 
print("\n[5] Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

 
# 6. CALLBACKS
 
print("\n[6] Setting up callbacks...")

# Learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    elif epoch < 30:
        return lr * 0.1
    else:
        return lr * 0.01

callbacks_list = [
    callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    callbacks.ModelCheckpoint('models/best_cnn_radioml.keras', 
                             monitor='val_accuracy', 
                             save_best_only=True,
                             verbose=1)
]

 
# 7. TRAIN MODEL
 
print("\n[7] Training CNN...")
print("    This will take 10-20 minutes depending on your hardware")

start_time = time.time()

history = model.fit(
    X_train, y_train,
    batch_size=256,
    epochs=50,
    validation_split=0.1,
    callbacks=callbacks_list,
    verbose=1
)

train_time = time.time() - start_time
print(f"\n    Training time: {train_time/60:.2f} minutes")

 
# 8. EVALUATE
 
print("\n[8] Evaluating model...")

# Overall accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"    Overall Accuracy: {test_acc*100:.2f}%")

# SNR subset evaluation
high_snr_mask = z_test >= 10
low_snr_mask = z_test < 0

if np.sum(high_snr_mask) > 0:
    loss_high, acc_high = model.evaluate(X_test[high_snr_mask], y_test[high_snr_mask], verbose=0)
    print(f"    High SNR (≥10dB): {acc_high*100:.2f}% ({np.sum(high_snr_mask)} samples)")

if np.sum(low_snr_mask) > 0:
    loss_low, acc_low = model.evaluate(X_test[low_snr_mask], y_test[low_snr_mask], verbose=0)
    print(f"    Low SNR (<0dB): {acc_low*100:.2f}% ({np.sum(low_snr_mask)} samples)")

 
# 9. PLOT HISTORY
 
print("\n[9] Saving training history plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
os.makedirs('docs', exist_ok=True)
plt.savefig('docs/cnn_training_history.png', dpi=150)
print("    Saved to: docs/cnn_training_history.png")

 
# 10. SAVE MODEL
 
print("\n[10] Saving model...")
os.makedirs('models', exist_ok=True)
model.save('models/cnn_radioml_final.keras')
print("    Model saved to: models/cnn_radioml_final.keras")

 
# 11. CONFUSION MATRIX
 
print("\n[11] Generating confusion matrix...")
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=modulations, yticklabels=modulations)
plt.title(f'CNN Confusion Matrix - RadioML\nAccuracy: {test_acc*100:.2f}%')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('docs/cnn_confusion_matrix.png', dpi=150)
print("    Saved to: docs/cnn_confusion_matrix.png")

print("\n" + "=" * 70)
print("CNN TRAINING COMPLETE!")
print("=" * 70)
print(f"\n[SUMMARY]")
print(f"    Overall Accuracy: {test_acc*100:.2f}%")
print(f"    High SNR (≥10dB): {acc_high*100:.2f}%" if np.sum(high_snr_mask) > 0 else "    High SNR: N/A")
print(f"    Low SNR (<0dB): {acc_low*100:.2f}%" if np.sum(low_snr_mask) > 0 else "    Low SNR: N/A")
print("\n[Next Steps]")
print("1. Compare with Random Forest (40.72%)")
print("2. Test on live RTL-SDR: python src/test_live_cnn.py")
print("3. Hyperparameter tuning for even better results")
print("=" * 70)