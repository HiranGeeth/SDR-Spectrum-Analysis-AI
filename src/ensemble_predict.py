# src/ensemble_predict.py
import numpy as np
import joblib
import tensorflow as tf
import json
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HYBRID ENSEMBLE: Random Forest + CNN")
print("=" * 70)


# 1. LOAD DATA
print("\n[1] Loading RadioML dataset...")
X = np.load('data/radioml_X.npy')
Y = np.load('data/radioml_Y.npy')  # String labels

with open('data/modulations.json', 'r') as f:
    class_names = json.load(f)

print(f"    Samples: {len(X)}")
print(f"    Classes: {len(class_names)}")

# Encode string labels to integers
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
num_classes = len(class_names)

print(f"    Encoded classes: {list(range(num_classes))}")

# 2. SPLIT DATA
print("\n[2] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
)
print(f"    Train: {len(X_train)} samples")
print(f"    Test: {len(X_test)} samples")

# 3. LOAD MODELS
print("\n[3] Loading trained models...")

# Load Random Forest (trained on flattened IQ)
rf = joblib.load('models/rf_radioml.pkl')
rf_scaler = joblib.load('models/scaler_radioml.pkl')
print(f"    Random Forest loaded")


print(f"    Random Forest classes: {rf.classes_[:5]}...")

if isinstance(rf.classes_[0], str):
    print("    Converting RF string labels to integers...")
    # Map string labels to integers using label_encoder
    rf_class_to_int = {cls: idx for idx, cls in enumerate(class_names)}
    rf_int_classes = [rf_class_to_int[cls] for cls in rf.classes_]
else:
    rf_int_classes = rf.classes_

# Load CNN
cnn = tf.keras.models.load_model('models/cnn_radioml_final.keras')
print(f"    CNN loaded")

# 4. PREDICTION FUNCTIONS
def get_rf_predictions(X_test):
    """Get RF predictions and probabilities as integers."""
    X_flat = X_test.reshape(X_test.shape[0], -1)
    X_scaled = rf_scaler.transform(X_flat)
    
    # Get predictions
    preds = rf.predict(X_scaled)
    
    # Convert string predictions to integers if needed
    if isinstance(preds[0], str):
        rf_class_to_int = {cls: idx for idx, cls in enumerate(class_names)}
        preds_int = np.array([rf_class_to_int[p] for p in preds])
    else:
        preds_int = preds
    
    # Get probabilities
    probs = rf.predict_proba(X_scaled)
    
    # If RF was trained on strings, we need to reorder probabilities
    if isinstance(rf.classes_[0], str):
        probs_reordered = np.zeros((len(probs), len(class_names)))
        for i, cls in enumerate(rf.classes_):
            cls_idx = class_names.index(cls)
            probs_reordered[:, cls_idx] = probs[:, i]
        probs = probs_reordered
    
    return preds_int, probs

def get_cnn_predictions(X_test):
    """Get CNN predictions and probabilities."""
    X_cnn = X_test.reshape(X_test.shape[0], 2, 128, 1)
    probs = cnn.predict(X_cnn, verbose=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs

# 5. EVALUATE INDIVIDUAL MODELS
print("\n[4] Individual model performance...")

rf_pred, rf_probs = get_rf_predictions(X_test)
cnn_pred, cnn_probs = get_cnn_predictions(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
cnn_acc = accuracy_score(y_test, cnn_pred)

print(f"    Random Forest: {rf_acc*100:.2f}%")
print(f"    CNN:           {cnn_acc*100:.2f}%")

# 6. ENSEMBLE STRATEGIES
print("\n[5] Testing ensemble strategies...")

# Strategy 1: Average Probabilities
avg_probs = (rf_probs + cnn_probs) / 2
pred_avg = np.argmax(avg_probs, axis=1)
acc_avg = accuracy_score(y_test, pred_avg)

# Strategy 2: Weighted Average (RF=0.3, CNN=0.7)
weighted_probs = (0.3 * rf_probs) + (0.7 * cnn_probs)
pred_weighted = np.argmax(weighted_probs, axis=1)
acc_weighted = accuracy_score(y_test, pred_weighted)

# Strategy 3: Majority Vote
pred_vote = []
for i in range(len(rf_pred)):
    if rf_pred[i] == cnn_pred[i]:
        pred_vote.append(rf_pred[i])
    else:
        # Use CNN when uncertain (it's slightly better)
        pred_vote.append(cnn_pred[i])
pred_vote = np.array(pred_vote)
acc_vote = accuracy_score(y_test, pred_vote)

# Strategy 4: Confidence-based Selection
pred_confidence = []
for i in range(len(rf_pred)):
    rf_conf = np.max(rf_probs[i])
    cnn_conf = np.max(cnn_probs[i])
    if cnn_conf > rf_conf:
        pred_confidence.append(cnn_pred[i])
    else:
        pred_confidence.append(rf_pred[i])
pred_confidence = np.array(pred_confidence)
acc_confidence = accuracy_score(y_test, pred_confidence)

print(f"    Average Probabilities:    {acc_avg*100:.2f}%")
print(f"    Weighted Average (CNN 70%): {acc_weighted*100:.2f}%")
print(f"    Majority Vote:            {acc_vote*100:.2f}%")
print(f"    Confidence Selection:     {acc_confidence*100:.2f}%")

# Find best
accuracies = {
    'RF Only': rf_acc,
    'CNN Only': cnn_acc,
    'Average Probabilities': acc_avg,
    'Weighted Average': acc_weighted,
    'Majority Vote': acc_vote,
    'Confidence Selection': acc_confidence
}
best_method = max(accuracies, key=accuracies.get)
best_acc = accuracies[best_method]

print(f"\n    Best: {best_method} ({best_acc*100:.2f}%)")

# 7. DETAILED ANALYSIS
print("\n[6] Detailed ensemble analysis...")

# Use best method for final predictions
if best_method == 'RF Only':
    final_pred = rf_pred
elif best_method == 'CNN Only':
    final_pred = cnn_pred
elif best_method == 'Average Probabilities':
    final_pred = pred_avg
elif best_method == 'Weighted Average':
    final_pred = pred_weighted
elif best_method == 'Majority Vote':
    final_pred = pred_vote
else:  # Confidence Selection
    final_pred = pred_confidence

print("\n    Classification Report:")
print(classification_report(y_test, final_pred, target_names=class_names))

# 8. ENSEMBLE IMPROVEMENT
print("\n[7] Ensemble improvement analysis...")

rf_correct = (rf_pred == y_test)
cnn_correct = (cnn_pred == y_test)
ensemble_correct = (final_pred == y_test)

both_correct = rf_correct & cnn_correct
both_wrong = ~rf_correct & ~cnn_correct
rf_only_correct = rf_correct & ~cnn_correct
cnn_only_correct = ~rf_correct & cnn_correct

total = len(y_test)

print(f"\n    Both correct:   {np.sum(both_correct)/total*100:.1f}% ({np.sum(both_correct)} samples)")
print(f"    Both wrong:     {np.sum(both_wrong)/total*100:.1f}% ({np.sum(both_wrong)} samples)")
print(f"    RF only correct:{np.sum(rf_only_correct)/total*100:.1f}% ({np.sum(rf_only_correct)} samples)")
print(f"    CNN only correct:{np.sum(cnn_only_correct)/total*100:.1f}% ({np.sum(cnn_only_correct)} samples)")

ensemble_fixed = ensemble_correct & ~rf_correct & ~cnn_correct
print(f"\n    Ensemble fixed: {np.sum(ensemble_fixed)} ")

# 9. SAVE
print("\n[8] Saving ensemble configuration...")

import os
os.makedirs('models', exist_ok=True)
ensemble_config = {
    'best_method': best_method,
    'accuracy': best_acc,
    'rf_accuracy': rf_acc,
    'cnn_accuracy': cnn_acc,
    'improvement': (best_acc - max(rf_acc, cnn_acc)),
    'class_names': class_names
}

with open('models/ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f, indent=2)

print("    Saved to: models/ensemble_config.json")

# 10. SUMMARY
print("\n" + "=" * 70)
print("ENSEMBLE EVALUATION COMPLETE! ")
print("=" * 70)
print(f"\n[SUMMARY]")
print(f"    RF Only:           {rf_acc*100:.2f}%")
print(f"    CNN Only:          {cnn_acc*100:.2f}%")
print(f"    Best Ensemble:     {best_acc*100:.2f}% ({best_method})")
print(f"    Improvement:       {(best_acc - max(rf_acc, cnn_acc))*100:.2f}%")
print("=" * 70)