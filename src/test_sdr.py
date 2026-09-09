# src/test_sdr.py - Simple RTL-SDR Test
import numpy as np
from rtlsdr import RtlSdr
import time
import matplotlib.pyplot as plt
from scipy import signal

print("=" * 70)
print("RTL-SDR TEST - SIMPLE READER")
print("=" * 70)

# 1. INITIALIZE SDR
print("\n[1] Initializing RTL-SDR...") 

try:
    sdr = RtlSdr()
    print("✅ RTL-SDR initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize RTL-SDR: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure your RTL-SDR is plugged in")
    print("2. Run Zadig and install WinUSB driver")
    print("3. Check if another program is using the SDR")
    exit(1)

# 2. CONFIGURE SDR
print("\n[2] Configuring RTL-SDR...")

# Set parameters
sdr.sample_rate = 2.4e6  # 2.4 MHz
sdr.center_freq = 100e6  # 100 MHz (FM broadcast band)
sdr.gain = 30  # Gain in dB (0-49.6)
#sdr.freq_correction = 0  # PPM correction

print(f"   Sample Rate: {sdr.sample_rate/1e6:.2f} MHz")
print(f"   Center Frequency: {sdr.center_freq/1e6:.2f} MHz")
print(f"   Gain: {sdr.gain} dB")
print(f"   Freq Correction: {sdr.freq_correction} ppm")

# 3. READ SAMPLES
print("\n[3] Reading samples...")

num_samples = 262144  # Number of samples to read

try:
    print(f"   Reading {num_samples} samples...")
    samples = sdr.read_samples(num_samples)
    print(f"   ✅ Read {len(samples)} samples")
    
except Exception as e:
    print(f"❌ Failed to read samples: {e}")
    sdr.close()
    exit(1)

# 4. ANALYZE SAMPLES
print("\n[4] Analyzing samples...")

# Basic statistics
real_part = np.real(samples)
imag_part = np.imag(samples)
magnitude = np.abs(samples)

print(f"   Real part: min={np.min(real_part):.4f}, max={np.max(real_part):.4f}, mean={np.mean(real_part):.4f}")
print(f"   Imag part: min={np.min(imag_part):.4f}, max={np.max(imag_part):.4f}, mean={np.mean(imag_part):.4f}")
print(f"   Magnitude: min={np.min(magnitude):.4f}, max={np.max(magnitude):.4f}, mean={np.mean(magnitude):.4f}")

# SNR estimate
signal_power = np.mean(magnitude ** 2)
noise_power = np.var(samples) / 2
snr_db = 10 * np.log10(signal_power / (noise_power + 1e-12))

print(f"   Signal Power: {signal_power:.6f}")
print(f"   Noise Power: {noise_power:.6f}")
print(f"   SNR: {snr_db:.2f} dB")

# 5. COMPUTE SPECTRUM
print("\n[5] Computing spectrum...")

# FFT
fft_data = np.fft.fftshift(np.fft.fft(samples))
fft_magnitude = np.abs(fft_data)
fft_freq = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/sdr.sample_rate))

# Convert to dB
fft_db = 20 * np.log10(fft_magnitude + 1e-12)

# Find peaks (potential signals)
peak_indices = signal.find_peaks(fft_db, height=np.max(fft_db) - 30, distance=20)[0]
peak_freqs = fft_freq[peak_indices] + sdr.center_freq
peak_dbs = fft_db[peak_indices]

print(f"   FFT size: {len(fft_data)}")
print(f"   Frequency range: {fft_freq.min()/1e6:.2f} to {fft_freq.max()/1e6:.2f} MHz")
print(f"   Found {len(peak_indices)} peaks above threshold")

# 6. PRINT SIGNAL DETECTION
print("\n[6] Signal detection:")

if len(peak_indices) > 0:
    print("   Potential signals detected at:")
    for freq, db in zip(peak_freqs[:5], peak_dbs[:5]):
        print(f"      {freq/1e6:.4f} MHz  ({db:.1f} dB)")
    if len(peak_indices) > 5:
        print(f"      ... and {len(peak_indices)-5} more")
else:
    print("   No strong signals detected. Try:")
    print("   - Changing frequency (try FM band: 88-108 MHz)")
    print("   - Adjusting gain (current: {sdr.gain} dB)")
    print("   - Using a different antenna")

# 7. PLOT
print("\n[7] Generating plots...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Time domain (first 1000 samples)
ax1.plot(real_part[:1000], 'b-', alpha=0.7, label='I')
ax1.plot(imag_part[:1000], 'r-', alpha=0.7, label='Q')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Amplitude')
ax1.set_title(f'IQ Samples (first 1000) - {sdr.center_freq/1e6:.2f} MHz')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Frequency domain
ax2.plot(fft_freq/1e6, fft_db, 'g-', alpha=0.7)
ax2.set_xlabel('Frequency offset (MHz)')
ax2.set_ylabel('Magnitude (dB)')
ax2.set_title(f'Spectrum - Center: {sdr.center_freq/1e6:.2f} MHz, SNR: {snr_db:.1f} dB')
ax2.grid(True, alpha=0.3)

# Mark peaks
for peak_idx in peak_indices[:10]:
    ax2.plot(fft_freq[peak_idx]/1e6, fft_db[peak_idx], 'r*', markersize=10)

plt.tight_layout()
plt.savefig('data/sdr_test_plot.png', dpi=150)
print(f"   ✅ Plot saved to: data/sdr_test_plot.png")

# 8. DISPLAY SUMMARY
print("\n" + "=" * 70)
print("SDR TEST COMPLETE! 🎉")
print("=" * 70)
print(f"\n[SUMMARY]")
print(f"   SDR: RTL-SDR Blog V3 (or compatible)")
print(f"   Center Frequency: {sdr.center_freq/1e6:.2f} MHz")
print(f"   Sample Rate: {sdr.sample_rate/1e6:.2f} MHz")
print(f"   SNR: {snr_db:.2f} dB")
print(f"   Signals Detected: {len(peak_indices)}")
print(f"   Plot saved to: data/sdr_test_plot.png")
print("\n[NEXT STEPS]")
print("1. Open: data/sdr_test_plot.png to see the spectrum")
print("2. Adjust the center frequency if needed")
print("3. Run the live scanner: streamlit run src/live_scanner.py")
print("=" * 70)

# 9. CLEANUP
sdr.close()
print("\n[INFO] SDR closed")