## Algorithmic Simulation & Golden Reference Model

### Goal
The Python(NumPy) algorithmic simulation serves as the bit-accurate golden reference model for our hardware accelerator. Before transitioning to RTL design, this simulation is used to:

1. **Validate Algorithmic Correctness:** Ensure the hardware-aware adaptations of Locality-Sensitive Hashing (LSH), counting-sort, and bucketed attention logically match the underlying mathematics of the Reformer architecture.
2. **Model Hardware Quantization:** Explicitly simulate the transition from floating-point arithmetic to 8-bit fixed-point (`int8`) embeddings and random projection matrices.
3. **Prevent Datapath Overflow:** Model the saturation limits of the 16-bit Multiply-Accumulate (MAC) accumulator (`int16`) to determine the necessary scaling factor (e.g., `scale = 15.0`) that guarantees no integer overflow occurs during the $QK^T$ stage.
4. **Establish a Verification Baseline:** Generate a ground-truth output matrix that the subsequent Verilog RTL and `cocotb` testbenches will be directly asserted against to prove functional correctness across different sequence lengths. 

### Simulation Result
Based on an input sequence length of $L=128$, an embedding dimension of $d=32$, and an 8-bit scaling factor of `15.0`, the simulation confirmed that the 16-bit serial datapath remains safely below the signed integer limit of `32,767`:

```text
Max value in 16-bit MAC Accumulator: 11160
Final Attention Output Shape: (128, 32)