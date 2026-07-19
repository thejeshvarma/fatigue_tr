"""
==========================================================================
  PYNQ On-Board Test: Fatigue Detection FPGA Accelerator
==========================================================================
  Copy this file + pynq_test_data/ + fatigue_overlay.bit + .hwh to PYNQ.
  Run as a Jupyter notebook on the board, or as a Python script.

  This script:
    1. Loads the FPGA overlay (bitstream)
    2. Discovers the HLS IP register map
    3. Runs all 194 test samples through the FPGA accelerator
    4. Compares FPGA results vs Keras reference predictions
    5. Reports accuracy, confusion matrix, and latency
==========================================================================
"""

import numpy as np
import time

# -----------------------------------------------------------------------
# 1. LOAD THE FPGA OVERLAY
# -----------------------------------------------------------------------
from pynq import Overlay

# Update this path to wherever you put the files on the PYNQ board
OVERLAY_PATH = "/home/xilinx/fatigue_overlay.bit"
TEST_DATA_DIR = "/home/xilinx/pynq_test_data"

print("Loading FPGA overlay...")
ol = Overlay(OVERLAY_PATH)
print("Overlay loaded!")

# List all IPs in the design
print("\nAvailable IPs:")
for name, details in ol.ip_dict.items():
    print(f"  {name}: {details['phys_addr']:#010x} - {details['type']}")

# -----------------------------------------------------------------------
# 2. ACCESS THE HLS ACCELERATOR
# -----------------------------------------------------------------------
# The IP name depends on your Vivado block design.
# Common names: 'myproject_0', 'myproject_0/control'
# Adjust if needed after checking ol.ip_dict above.

nn = ol.myproject_0  # <-- adjust this name if different

print("\nRegister Map:")
print(nn.register_map)

# -----------------------------------------------------------------------
# 3. FIXED-POINT CONVERSION HELPERS
# -----------------------------------------------------------------------
# Your model uses ap_fixed<16,6>:
#   - 16 total bits, 6 integer bits, 10 fractional bits
#   - Range: [-32.0, +31.999...]
#   - Resolution: 1/1024 ~ 0.000977

TOTAL_BITS = 16
INT_BITS = 6
FRAC_BITS = TOTAL_BITS - INT_BITS  # 10

def float_to_apfixed(val):
    """Convert float to ap_fixed<16,6> as a 16-bit unsigned integer for register write."""
    scaled = int(round(val * (1 << FRAC_BITS)))
    # Clamp to representable range
    max_val = (1 << (TOTAL_BITS - 1)) - 1   # 32767
    min_val = -(1 << (TOTAL_BITS - 1))       # -32768
    scaled = max(min_val, min(max_val, scaled))
    # Two's complement for negative values
    if scaled < 0:
        scaled = (1 << TOTAL_BITS) + scaled
    return scaled & 0xFFFF

def apfixed_to_float(val):
    """Convert ap_fixed<16,6> register value back to float."""
    val = val & 0xFFFF
    if val >= (1 << (TOTAL_BITS - 1)):
        val -= (1 << TOTAL_BITS)
    return val / (1 << FRAC_BITS)

# -----------------------------------------------------------------------
# 4. DISCOVER REGISTER OFFSETS
# -----------------------------------------------------------------------
# After HLS synthesis with s_axilite, Vitis HLS generates a driver with
# register offsets. You need to find them from either:
#   a) nn.register_map (printed above)
#   b) The xmyproject_hw.h header file from HLS export
#
# Typical layout for s_axilite with arrays:
#   0x00 : Control (ap_start=bit0, ap_done=bit1, ap_idle=bit2)
#   0x10+: input_1[0..9]  (each element = 4 bytes apart, 16-bit data in lower bits)
#   0x40+: layer9_out[0..2] (each element = 4 bytes apart)
#
# !! IMPORTANT: Replace these with actual values from your register_map !!

CTRL_REG   = 0x00       # Control register
AP_START   = 0x01       # Bit 0 of CTRL
AP_DONE    = 0x02       # Bit 1 of CTRL
AP_IDLE    = 0x04       # Bit 2 of CTRL

# Verified AXI-lite offsets from xmyproject_hw.h
INPUT_BASE  = 0x10      # Start of input_1_0
INPUT_STEP  = 0x08      # Bytes between input registers (0x10, 0x18, 0x20...)
OUTPUT_BASE = 0x60      # Start of layer9_out_0
OUTPUT_STEP = 0x10      # Bytes between output registers (0x60, 0x70, 0x80...)

# Uncomment this to auto-discover from register_map:
# print("\nFull register map details:")
# for attr_name in dir(nn.register_map):
#     if not attr_name.startswith('_'):
#         print(f"  {attr_name}: {getattr(nn.register_map, attr_name)}")

# -----------------------------------------------------------------------
# 5. SINGLE INFERENCE FUNCTION
# -----------------------------------------------------------------------
def fpga_inference(features_10):
    """
    Run one inference on the FPGA.
    
    Args:
        features_10: numpy array of 10 float values (already preprocessed)
    Returns:
        numpy array of 3 float values (softmax probabilities)
    """
    # Write 10 input features
    for i in range(10):
        nn.write(INPUT_BASE + i * INPUT_STEP, float_to_apfixed(features_10[i]))
    
    # Start accelerator
    nn.write(CTRL_REG, AP_START)
    
    # Wait for done (poll AP_DONE bit)
    while (nn.read(CTRL_REG) & AP_DONE) == 0:
        pass
    
    # Read 3 output probabilities
    results = np.zeros(3, dtype=np.float32)
    for i in range(3):
        raw = nn.read(OUTPUT_BASE + i * OUTPUT_STEP)
        results[i] = apfixed_to_float(raw)
    
    return results

# -----------------------------------------------------------------------
# 6. LOAD TEST DATA
# -----------------------------------------------------------------------
print("\nLoading test data...")
test_inputs = np.load(f"{TEST_DATA_DIR}/test_inputs.npy")
test_labels = np.load(f"{TEST_DATA_DIR}/test_labels.npy")
keras_preds = np.load(f"{TEST_DATA_DIR}/keras_preds.npy")

# Load class names
class_names = {}
with open(f"{TEST_DATA_DIR}/class_names.txt") as f:
    for line in f:
        idx, name = line.strip().split(" ", 1)
        class_names[int(idx)] = name

n_samples = len(test_inputs)
print(f"  Samples  : {n_samples}")
print(f"  Features : {test_inputs.shape[1]}")
print(f"  Classes  : {class_names}")

# -----------------------------------------------------------------------
# 7. RUN ALL TEST SAMPLES THROUGH FPGA
# -----------------------------------------------------------------------
print(f"\nRunning {n_samples} inferences on FPGA...")

fpga_preds_raw = np.zeros((n_samples, 3), dtype=np.float32)
fpga_classes = np.zeros(n_samples, dtype=np.int32)
keras_classes = np.argmax(keras_preds, axis=1)

start_time = time.time()

for i in range(n_samples):
    fpga_preds_raw[i] = fpga_inference(test_inputs[i])
    fpga_classes[i] = np.argmax(fpga_preds_raw[i])
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{n_samples}...")

total_time = time.time() - start_time
avg_latency_ms = (total_time / n_samples) * 1000

print(f"\nDone! Total time: {total_time:.3f}s, Avg latency: {avg_latency_ms:.3f}ms/sample")

# -----------------------------------------------------------------------
# 8. COMPUTE ACCURACY METRICS
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FPGA ON-BOARD TEST RESULTS")
print("=" * 60)

# FPGA vs Ground Truth
fpga_correct = np.sum(fpga_classes == test_labels)
fpga_accuracy = fpga_correct / n_samples

# FPGA vs Keras (quantisation agreement)
fpga_vs_keras = np.sum(fpga_classes == keras_classes)
fpga_keras_agreement = fpga_vs_keras / n_samples

# Keras vs Ground Truth (reference)
keras_correct = np.sum(keras_classes == test_labels)
keras_accuracy = keras_correct / n_samples

print(f"\n  Keras  accuracy (reference)    : {keras_accuracy:.4f}  ({keras_correct}/{n_samples})")
print(f"  FPGA   accuracy (vs truth)     : {fpga_accuracy:.4f}  ({fpga_correct}/{n_samples})")
print(f"  FPGA-Keras agreement           : {fpga_keras_agreement:.4f}  ({fpga_vs_keras}/{n_samples})")
print(f"  Avg inference latency          : {avg_latency_ms:.3f} ms")

# Per-class breakdown
print(f"\n  Per-class accuracy (FPGA vs Ground Truth):")
for cls_idx, cls_name in class_names.items():
    mask = test_labels == cls_idx
    if mask.sum() > 0:
        cls_acc = np.sum(fpga_classes[mask] == test_labels[mask]) / mask.sum()
        print(f"    {cls_name:12s}: {cls_acc:.4f}  ({np.sum(fpga_classes[mask] == test_labels[mask])}/{mask.sum()})")

# Confusion Matrix
print(f"\n  Confusion Matrix (rows=true, cols=FPGA predicted):")
n_classes = len(class_names)
cm = np.zeros((n_classes, n_classes), dtype=np.int32)
for true, pred in zip(test_labels, fpga_classes):
    cm[true, pred] += 1

header = "          " + "  ".join(f"{class_names[i]:>8s}" for i in range(n_classes))
print(f"  {header}")
for i in range(n_classes):
    row = f"  {class_names[i]:>8s}" + "  ".join(f"{cm[i,j]:>8d}" for j in range(n_classes))
    print(row)

# Quantisation error analysis
print(f"\n  Quantisation Error (FPGA vs Keras softmax outputs):")
errors = np.abs(fpga_preds_raw - keras_preds)
print(f"    Mean absolute error : {errors.mean():.6f}")
print(f"    Max  absolute error : {errors.max():.6f}")
print(f"    Std  absolute error : {errors.std():.6f}")

# Show mismatches if any
mismatches = np.where(fpga_classes != keras_classes)[0]
if len(mismatches) > 0:
    print(f"\n  Quantisation Mismatches ({len(mismatches)} samples where FPGA != Keras):")
    for idx in mismatches[:10]:  # show first 10
        print(f"    Sample {idx}: Keras={class_names[keras_classes[idx]]}, "
              f"FPGA={class_names[fpga_classes[idx]]}, "
              f"Truth={class_names[test_labels[idx]]}")
    if len(mismatches) > 10:
        print(f"    ... and {len(mismatches) - 10} more")
else:
    print(f"\n  ** Perfect agreement: FPGA matches Keras on all {n_samples} samples! **")

print("\n" + "=" * 60)
