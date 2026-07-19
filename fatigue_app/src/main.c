#include <stdio.h>
#include "xil_printf.h"
#include "xparameters.h"
#include "xmyproject.h"

// ─── FLOAT TO AP_FIXED<16,6> CONVERSION HELPERS ──────────────────────
// ap_fixed<16,6> has 10 fractional bits. Scale factor is 2^10 = 1024.
// Range: [-32.0, 31.999...]
#define FIXED_SCALE 1024.0f

u32 float_to_fixed16_6(float val) {
    int32_t scaled = (int32_t)(val * FIXED_SCALE + (val >= 0.0f ? 0.5f : -0.5f));
    // Clamp to 16-bit signed integer limits
    if (scaled > 32767)  scaled = 32767;
    if (scaled < -32768) scaled = -32768;
    return (u32)(scaled & 0xFFFF);
}

float fixed16_6_to_float(u32 reg_val) {
    int16_t signed_val = (int16_t)(reg_val & 0xFFFF);
    return (float)signed_val / FIXED_SCALE;
}

// ─── TEST SCENARIOS ──────────────────────────────────────────────────
// 5 test samples (each has 10 scaled+selected features)
#define NUM_TESTS 5
float test_samples[NUM_TESTS][10] = {
    { 0.5f, -0.3f,  1.2f,  0.8f, -0.1f,  0.6f, -0.9f,  0.4f,  1.1f, -0.2f}, // Alert sample
    {-0.2f,  0.7f,  0.3f,  1.5f,  0.2f,  0.1f, -0.4f,  0.9f,  0.0f,  0.6f}, // Moderate sample
    { 1.1f, -0.8f,  0.5f,  0.2f,  1.3f,  0.4f, -0.6f,  0.3f,  0.7f, -0.1f}, // Severe sample
    {-0.5f,  0.4f,  0.9f,  0.1f, -0.3f,  1.2f,  0.8f,  0.5f,  0.2f,  0.3f}, 
    { 0.3f,  1.0f,  0.1f,  0.6f,  0.7f,  0.2f,  0.4f,  1.1f,  0.5f,  0.8f}
};

const char* class_names[] = {"Alert", "Moderate", "Severe"};

int main() {
    XMyproject ip_inst;
    XMyproject_Config *cfg_ptr;
    int status;

    printf("\r\n======================================================\r\n");
    printf("   Fatigue Detection Neural Network FPGA Accelerator\r\n");
    printf("======================================================\r\n\r\n");

    // Initialize the HLS IP
#ifdef SDT
    // System Device Tree flow (Vitis Unified default)
    // Find the base address from xparameters.h
    // Replace XPAR_MYPROJECT_0_BASEADDR if your block design has a different name
    #ifdef XPAR_MYPROJECT_0_BASEADDR
        cfg_ptr = XMyproject_LookupConfig(XPAR_MYPROJECT_0_BASEADDR);
    #else
        #error "XPAR_MYPROJECT_0_BASEADDR not defined in xparameters.h. Check your hardware design name."
    #endif
#else
    // Legacy Device ID flow
    #ifdef XPAR_MYPROJECT_0_DEVICE_ID
        cfg_ptr = XMyproject_LookupConfig(XPAR_MYPROJECT_0_DEVICE_ID);
    #else
        // Fallback to searching by base address
        cfg_ptr = XMyproject_LookupConfig(XPAR_MYPROJECT_0_BASEADDR);
    #endif
#endif

    if (cfg_ptr == NULL) {
        printf("ERROR: Could not find HLS IP configuration.\r\n");
        return -1;
    }

    status = XMyproject_CfgInitialize(&ip_inst, cfg_ptr);
    if (status != XST_SUCCESS) {
        printf("ERROR: HLS IP initialization failed.\r\n");
        return -1;
    }

    printf("HLS IP initialized successfully!\r\n\r\n");

    // Run inference for each test sample
    for (int s = 0; s < NUM_TESTS; s++) {
        printf("--- Running Test Sample %d ---\r\n", s + 1);
        
        // Write the 10 input registers
        XMyproject_Set_input_1_0(&ip_inst, float_to_fixed16_6(test_samples[s][0]));
        XMyproject_Set_input_1_1(&ip_inst, float_to_fixed16_6(test_samples[s][1]));
        XMyproject_Set_input_1_2(&ip_inst, float_to_fixed16_6(test_samples[s][2]));
        XMyproject_Set_input_1_3(&ip_inst, float_to_fixed16_6(test_samples[s][3]));
        XMyproject_Set_input_1_4(&ip_inst, float_to_fixed16_6(test_samples[s][4]));
        XMyproject_Set_input_1_5(&ip_inst, float_to_fixed16_6(test_samples[s][5]));
        XMyproject_Set_input_1_6(&ip_inst, float_to_fixed16_6(test_samples[s][6]));
        XMyproject_Set_input_1_7(&ip_inst, float_to_fixed16_6(test_samples[s][7]));
        XMyproject_Set_input_1_8(&ip_inst, float_to_fixed16_6(test_samples[s][8]));
        XMyproject_Set_input_1_9(&ip_inst, float_to_fixed16_6(test_samples[s][9]));

        // Start the hardware accelerator
        XMyproject_Start(&ip_inst);

        // Wait for completion (poll done status)
        while (!XMyproject_IsDone(&ip_inst));

        // Read outputs
        u32 raw_out0 = XMyproject_Get_layer9_out_0(&ip_inst);
        u32 raw_out1 = XMyproject_Get_layer9_out_1(&ip_inst);
        u32 raw_out2 = XMyproject_Get_layer9_out_2(&ip_inst);

        // Convert back to floats
        float p0 = fixed16_6_to_float(raw_out0);
        float p1 = fixed16_6_to_float(raw_out1);
        float p2 = fixed16_6_to_float(raw_out2);

        // Compute predicted class (argmax)
        int pred_class = 0;
        float max_p = p0;
        if (p1 > max_p) {
            max_p = p1;
            pred_class = 1;
        }
        if (p2 > max_p) {
            max_p = p2;
            pred_class = 2;
        }

        // Print results
        printf("  Probabilities: [Alert: %.4f, Moderate: %.4f, Severe: %.4f]\r\n", p0, p1, p2);
        printf("  Predicted State: %s\r\n\r\n", class_names[pred_class]);
    }

    printf("================ Testing Complete =================\r\n");

    return 0;
}
