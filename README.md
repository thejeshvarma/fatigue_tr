# Fatigue Detection FPGA Accelerator
This repository contains the hardware, firmware, and software source files to deploy a Fatigue Detection neural network model onto a Xilinx Zynq-7000 SoC (Zedboard) using `hls4ml`, Xilinx Vivado, and Vitis Unified IDE.

---

## 🚀 High-Level Implementation Flow

Follow these four phases sequentially when setting up the project on a new system:

### Phase 1: High-Level Synthesis (Vitis HLS)
*   **Goal**: Compile the neural network model into a hardware IP block.
*   **Action**:
    1. Open the HLS project in the `hls4ml_prj/` directory.
    2. Run High-Level Synthesis.
    3. Export the design to generate the packaged hardware IP block (`.zip` format).

### Phase 2: Hardware Integration (Vivado)
*   **Goal**: Connect the neural network accelerator to the Zynq ARM processor.
*   **Action**:
    1. Open the Vivado project in the `vivado_project/` directory.
    2. In the Block Design (`design_1.bd`):
        *   Upgrade the neural network IP block to use your newly synthesized version.
        *   Double-click the Zynq Processing System block, open **Clock Configuration > PL Fabric Clocks**, and ensure **`FCLK_CLK0`** is enabled (checked).
        *   Ensure the Processor System Reset block's **`dcm_locked`** pin is connected to a **Constant 1** block (to prevent the hardware from locking in reset).
    3. Run **Generate Bitstream** to compile the FPGA hardware.
    4. **Export the Hardware** (including the bitstream) to generate the updated system specification file (`fatigue.xsa`).

### Phase 3: Software Development (Vitis Unified IDE)
*   **Goal**: Create a standalone C program to control the FPGA accelerator from the processor.
*   **Action**:
    1. Open Vitis Unified IDE and point to an empty workspace.
    2. Create a **Platform Component** from your exported `fatigue.xsa` file.
    3. Create an empty standalone **Application Component** targeting the `ps7_cortexa9_0` core.
    4. Copy [main.c](fatigue_app/src/main.c) into your application source folder.
    5. Build the platform and application to compile the executable binary (`fatigue_app.elf`).

### Phase 4: Board Deployment & Execution (Zedboard)
*   **Goal**: Load the design onto the board and view prediction outputs.
*   **Action**:
    1. Connect the Zedboard JTAG and UART ports to your PC and power it on.
    2. Open your serial console (such as PuTTY or Tera Term) configured for COM port at **115200 baud** (with Flow control: None).
    3. Using Vitis or the XSDB console:
        *   Load the bitstream onto the FPGA.
        *   Initialize the processor.
        *   Download and execute the application binary.
    4. Watch the classification predictions print out on the serial terminal!

---

## 🛠️ Key Optimizations Implemented

*   **Softmax Lookup Table Fix**: Patched `nnet_activation.h` to force table arrays to be `static` and added `#pragma HLS inline` to the initialization functions. This allowed the compiler to calculate values at compile-time and fold them into ROM (BRAM), reducing Flip-Flop (FF) utilization by **95%** (from 595% down to **31%**) and cutting synthesis time from hours to **2.5 minutes**.
*   **DSPs Reuse Factor**: Increased `ReuseFactor` to `2` on the first two dense layers in `parameters.h` to share multipliers, bringing DSP utilization down to **80%** (178 / 220) to comfortably fit on the Zynq-7020 chip.

---

## 📂 Repository Layout

*   `hls4ml_prj/`: Contains the optimized HLS model files, layers, weights, and compilation TCL scripts.
*   `vivado_project/`: Contains the block design sources and the `.xpr` project file.
*   `fatigue_app/src/`: Contains the C driver source code (`main.c`) and linker scripts.
*   `pynq_onboard_test.py`: Python overlay test script if running PYNQ on board.
*   `scaler.pkl`, `feature_selector.pkl`, `fatigue_model_fpga.keras`: Machine learning model artifacts.
