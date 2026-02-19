import numpy as np

# --- Quantization Helpers ---
def quantize_int8(x, scale):
    """Quantizes a float array to 8-bit signed integer (range -128 to 127)."""
    # Rounding and clipping to simulate hardware register limits
    return np.clip(np.round(x * scale), -128, 127).astype(np.int8)

def saturate_int16(x):
    """Simulates a 16-bit accumulator with saturation logic to prevent overflow."""
    return np.clip(x, -32768, 32767).astype(np.int16)

# --- Hardware Stage Simulations ---

def stage1_lsh_hashing_quantized(X_q, R_q):
    """
    Simulates Stage 1: LSH Hashing with 8-bit inputs and a 16-bit MAC.
    X_q: (L, d) int8
    R_q: (d, num_hashes/2) int8
    """
    # Cast to int16 before multiplication to represent the 16-bit MAC accumulator [cite: 562]
    X_mac = X_q.astype(np.int16)
    R_mac = R_q.astype(np.int16)
    
    # Compute dot product and saturate at 16 bits
    projected_acc16 = saturate_int16(np.dot(X_mac, R_mac))
    
    # Concatenate [xR; -xR] to capture angular slices [cite: 564]
    concat_proj = np.concatenate([projected_acc16, -projected_acc16], axis=-1)
    
    # Argmax determines the bucket ID [cite: 564]
    bucket_ids = np.argmax(concat_proj, axis=-1)
    return bucket_ids

def stage2_bucket_sort_quantized(X_q, bucket_ids, chunk_size):
    """
    Simulates Stage 2: Counting Sort and Chunk Formation on 8-bit data.
    """
    L = X_q.shape[0]
    
    # Hardware computes these sorted indices to scatter via SPI [cite: 567]
    sort_idx = np.argsort(bucket_ids)
    sorted_X_q = X_q[sort_idx]
    
    # Logically divide into fixed-size chunks [cite: 568]
    num_chunks = L // chunk_size
    chunked_X_q = sorted_X_q.reshape(num_chunks, chunk_size, -1)
    
    return chunked_X_q, sort_idx

def stage3_chunked_dot_product_quantized(chunked_X_q):
    """
    Simulates Stage 3: Chunked QK^T Engine with 16-bit accumulation.
    """
    num_chunks, chunk_size, d = chunked_X_q.shape
    raw_scores_acc16 = np.zeros((num_chunks, chunk_size, chunk_size), dtype=np.int16)
    
    for c in range(num_chunks):
        # Cast to 16-bit to simulate the shared serial MAC datapath [cite: 569, 570]
        Q_chunk = chunked_X_q[c].astype(np.int16)
        K_chunk = chunked_X_q[c].astype(np.int16) 
        
        # Accumulate dot products and saturate
        raw_scores_acc16[c] = saturate_int16(np.dot(Q_chunk, K_chunk.T))
        
    return raw_scores_acc16

def host_post_processing(raw_scores_acc16, chunked_X_q, sort_idx, scale):
    """
    Simulates the RP2040 host un-sorting and computing Softmax[cite: 543, 579].
    """
    # Dequantize scores back to float for the host's softmax math
    # Note: Q * K means the scale is squared!
    dequantized_scores = raw_scores_acc16.astype(np.float32) / (scale ** 2)
    
    # 1. Softmax Normalization
    max_scores = np.max(dequantized_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(dequantized_scores - max_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 2. Value-Weighted Summation (Dequantizing V/X_chunk first)
    chunked_X_float = chunked_X_q.astype(np.float32) / scale
    attention_out_chunked = np.matmul(attention_weights, chunked_X_float)
    
    # 3. Un-sort to original sequence order [cite: 543]
    L = len(sort_idx)
    attention_out_flat = attention_out_chunked.reshape(L, -1)
    final_out = np.zeros_like(attention_out_flat)
    final_out[sort_idx] = attention_out_flat
    
    return final_out

# --- Testbench Configuration ---
L = 128             # Sequence length 
d = 32              # Embedding dimension 
num_hashes = 4      # Number of LSH buckets 
chunk_size = 16     # Tokens per chunk 
scale = 15.0        # Chosen quantization scaling factor

# Initialize random floating-point token vectors and projection matrix
X_float = np.random.randn(L, d)
R_float = np.random.randn(d, num_hashes // 2)

# Quantize to 8-bit for the ASIC 
X_q = quantize_int8(X_float, scale)
R_q = quantize_int8(R_float, scale)

# Execute Quantized Pipeline
bucket_ids = stage1_lsh_hashing_quantized(X_q, R_q)
chunked_X_q, sort_idx = stage2_bucket_sort_quantized(X_q, bucket_ids, chunk_size)
raw_scores_acc16 = stage3_chunked_dot_product_quantized(chunked_X_q)

# Host processes the final output
final_attention = host_post_processing(raw_scores_acc16, chunked_X_q, sort_idx, scale)

print(f"Max value in 16-bit MAC Accumulator: {np.max(np.abs(raw_scores_acc16))}")
print(f"Final Attention Output Shape: {final_attention.shape}")