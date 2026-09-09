# load_radioml.py - Load RadioML 2016.10A dataset correctly
import pickle
import numpy as np
import os
from pathlib import Path
import json

print("=" * 70)
print("LOADING RADIOML 2016.10A DATASET")
print("=" * 70)

def load_radioml():
    """Load RadioML dataset from pickle file"""
    
    # Find the dataset file
    dataset_path = Path("C:/Users/GeethD/.cache/kagglehub/datasets/nolasthitnotomorrow/radioml2016-deepsigcom/versions/1")
    pkl_file = dataset_path / "RML2016.10a_dict.pkl"
    
    if not pkl_file.exists():
        print(f"\n[ERROR] File not found at: {pkl_file}")
        return None
    
    print(f"\n[1] Loading from: {pkl_file}")
    print(f"    Size: {pkl_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Load pickle
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    print(f"\n[2] Dataset loaded!")
    print(f"    Type: {type(data)}")
    print(f"    Number of keys: {len(data)}")
    
    # Sample a few keys
    sample_keys = list(data.keys())[:5]
    print(f"    Sample keys: {sample_keys}")
    
    # Get all unique modulations and SNRs
    modulations = sorted(set([key[0] for key in data.keys()]))
    snrs = sorted(set([key[1] for key in data.keys()]))
    
    print(f"\n[3] Dataset info:")
    print(f"    Modulations: {modulations}")
    print(f"    Number of modulations: {len(modulations)}")
    print(f"    SNRs: {snrs}")
    print(f"    SNR range: {min(snrs)}dB to {max(snrs)}dB")
    
    # Check shape of one sample
    first_key = list(data.keys())[0]
    sample = data[first_key]
    print(f"\n[4] Sample shape: {sample.shape}")
    print(f"    (samples per SNR: {sample.shape[0]}, IQ channels: {sample.shape[1]}, symbols: {sample.shape[2]})")
    
    # Build X, Y, Z arrays
    print("\n[5] Building feature matrix...")
    
    X_list = []
    Y_list = []
    Z_list = []
    
    for (mod, snr), samples in data.items():
        # samples shape: (1000, 2, 128)
        for i in range(samples.shape[0]):
            X_list.append(samples[i])  # (2, 128)
            Y_list.append(mod)
            Z_list.append(snr)
    
    X = np.array(X_list)  # (n_samples, 2, 128)
    Y = np.array(Y_list)  # (n_samples,)
    Z = np.array(Z_list)  # (n_samples,)
    
    print(f"\n[6] Feature matrix built:")
    print(f"    X shape: {X.shape}")
    print(f"    Y shape: {Y.shape}")
    print(f"    Z shape: {Z.shape}")
    
    # Get modulation names
    modulation_names = sorted(set(Y))
    print(f"    Modulations: {modulation_names}")
    
    # Count samples per modulation
    print("\n[7] Class distribution:")
    for mod in modulation_names:
        count = np.sum(Y == mod)
        print(f"    {mod:10}: {count:6} samples")
    
    # Save as numpy arrays for faster loading
    print("\n[8] Saving as numpy arrays...")
    os.makedirs('data', exist_ok=True)
    np.save('data/radioml_X.npy', X)
    np.save('data/radioml_Y.npy', Y)
    np.save('data/radioml_SNR.npy', Z)
    
    # Save modulation names
    with open('data/modulations.json', 'w') as f:
        json.dump(modulation_names, f)
    
    print("    Saved to: data/")
    print("      - radioml_X.npy (IQ samples)")
    print("      - radioml_Y.npy (modulation labels)")
    print("      - radioml_SNR.npy (SNR values)")
    print("      - modulations.json (modulation names)")
    
    print("\n" + "=" * 70)
    print("LOADING COMPLETE!")
    print("=" * 70)
    print("\n[Next Steps]")
    print("1. Run: python src/train_radioml.py")
    print("2. Or explore: python src/explore_radioml.py")
    
    return X, Y, Z, modulation_names

if __name__ == "__main__":
    load_radioml()