import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    batch_size, seq_len, d_model = Q.shape

    # Dimension per head                       
    d_k = d_model//num_heads

    # Linear projections for Q, K, V                       
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    # Splitting into multiple heads                      
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k)

    # Transpose by swapping num_heads with seq_length for parallel processing                       
    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)

    # Scaled dot-product attention for each head                       
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2))

    # Scaling for stable range of values                       
    scaled_scores = scores / np.sqrt(d_k)

    # Apply softmax to get attention weights                       
    attention_weights = softmax(scaled_scores, axis = -1)

    # Weighted average for all Value vectors for each head                     
    attention_output = np.matmul(attention_weights, V_heads)

    # Transpose back to [batch_size, seq_len, num_heads, d_k] by swapping seq_len with num_heads                       
    attention_output = attention_output.transpose(0, 2, 1, 3)

     # Concatenating all heads                      
    concat_output = attention_output.reshape(batch_size, seq_len, d_model)

    # Apply the output projection for information sharing among all heads                       
    result = np.matmul(concat_output, W_o)

    return result
