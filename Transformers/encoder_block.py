import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """

    # Compute mean
    mean = np.mean(x, axis = -1, keepdims = True)

    # Compute variance
    variance = np.var(x, axis = -1, keepdims = True)

    # Normalization for numerical stability
    normalized = (x - mean) / np.sqrt(variance + eps)

    # Apply learnable parameters: scale (gamma) and shift (beta)
    result = gamma * normalized + beta

    return result

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """

    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projection of Q, K and V                      
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)
                           
    # Split into multiple heads
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k)
                           
    # Transpose for dot-product
    Q_heads = Q_heads.transpose(0, 2, 1, 3)
    K_heads = K_heads.transpose(0, 2, 1, 3)
    V_heads = V_heads.transpose(0, 2, 1, 3)

    # Compute attention scores for each head                      
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2))

    # Scale for stability                  
    scaled_scores = scores / np.sqrt(d_k)

    # Apply softmax                       
    attention_weights = softmax(scaled_scores, axis = -1)

    # Apply attention weights to values                       
    attention_out = np.matmul(attention_weights, V_heads)

    # Transpose back                       
    attention_out = attention_out.transpose(0, 2, 1, 3)

    # Concatenate the outputs of all heads                      
    concat = attention_out.reshape(batch_size, seq_len, d_model)

    # Apply output projection                       
    result = np.matmul(concat, W_o)

    return result
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """

    # First linear layer               
    hidden_layer = np.matmul(x, W1) + b1

    # Apply ReLU activation               
    relu_act = np.maximum(0, hidden_layer)

    # Second linear layer               
    result = np.matmul(relu_act, W2) + b2

    return result

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """

    # Compute Multi-head attention                
    sub_layer1 = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)

    # Apply Residual connection to sub_layer1               
    residual1 = x + sub_layer1

    # Apply layer normalization                
    norm1 = layer_norm(residual1, gamma1, beta1)

    # Compute feed-forward network                
    sub_layer2 = feed_forward(norm1, W1, b1, W2, b2)

    # Apply Residual connection to sub_layer2                
    residual2 = norm1 + sub_layer2

    # Apply layer normalization                 
    norm2 = layer_norm(residual2, gamma2, beta2)
    
    return norm2

    
