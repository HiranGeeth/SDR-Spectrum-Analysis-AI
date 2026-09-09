# src/train_radioml.py - Train on RadioML dataset
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("=" * 70)
print("RADIOML 2016.10A - SIGNAL CLASSIFICATION")
print("=" * 70)

# 1. Load data
print("\n[1] Loading dataset...")
X = np.load('data/radioml_X.npy')
Y = np.load('data/radioml_Y.npy')
Z = np.load('data/radioml_SNR.npy')

with open('data/modulations.json', 'r') as f:
    modulations = json.load(f)

print(f"    Samples: {len(X)}")
print(f"    IQ shape: {X.shape[1:]}")
print(f"    Modulations: {len(modulations)}")
print(f"    SNR range: {np.min(Z):.0f}dB to {np.max(Z):.0f}dB")

# 2. Prepare features (flatten IQ from [2, 128] to [256])
print("\n[2] Preparing features...")
X_flat = X.reshape(X.shape[0], -1)  # Shape: (n_samples, 256)
print(f"    Feature shape: {X_flat.shape}")

# 3. Split
print("\n[3] Splitting data...")
X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
    X_flat, Y, Z, test_size=0.2, random_state=42, stratify=Y
)
print(f"    Train: {len(X_train)} samples")
print(f"    Test: {len(X_test)} samples")

# 4. Scale
print("\n[4] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train Random Forest
print("\n[5] Training Random Forest...")
start_time = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
train_time = time.time() - start_time
print(f"    Training time: {train_time:.2f} seconds")

# 6. Evaluate
print("\n[6] Evaluating model...")
y_pred = rf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"    Accuracy: {accuracy*100:.2f}%")

print("\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=modulations))

# 7. Confusion Matrix
print("\n[7] Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=modulations, yticklabels=modulations)
plt.title(f'Confusion Matrix - RF Classifier\nAccuracy: {accuracy*100:.2f}%')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('docs/radioml_confusion_matrix.png', dpi=150)
print("    Saved to: docs/radioml_confusion_matrix.png")

# 8. Save model
print("\n[8] Saving model...")
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/rf_radioml.pkl')
joblib.dump(scaler, 'models/scaler_radioml.pkl')
print("    Model saved to: models/rf_radioml.pkl")
print("    Scaler saved to: models/scaler_radioml.pkl")

# 9. Test on high SNR vs low SNR
print("\n[9] Testing on SNR subsets...")
high_snr_mask = z_test >= 10
low_snr_mask = z_test < 0

if np.sum(high_snr_mask) > 0:
    y_pred_high = rf.predict(X_test_scaled[high_snr_mask])
    acc_high = accuracy_score(y_test[high_snr_mask], y_pred_high)
    print(f"    High SNR (≥10dB): {acc_high*100:.2f}% ({np.sum(high_snr_mask)} samples)")

if np.sum(low_snr_mask) > 0:
    y_pred_low = rf.predict(X_test_scaled[low_snr_mask])
    acc_low = accuracy_score(y_test[low_snr_mask], y_pred_low)
    print(f"    Low SNR (<0dB): {acc_low*100:.2f}% ({np.sum(low_snr_mask)} samples)")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)