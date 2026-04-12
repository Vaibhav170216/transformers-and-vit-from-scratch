import numpy as np

def gelu(x):
  
    # GELU activation function
    return 0.5 * x * (1 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis = -1):
  
    # Softmax function
    e_x = np.exp(x - np.max(x, axis = axis, keepdims = True))
  
    return e_x / np.sum(e_x, axis = axis, keepdims = True)

def layer_norm(x, gamma, beta, eps = 1e-6):

    # Calculate mean
    mean = np.mean(x, axis = -1, keepdims = True)
    # Calculate variance
    var = np.var(x, axis = -1, keepdims = True)
    # Normalize the input
    normalized_x = (x - mean) / np.sqrt(var + eps)
    # Apply gamma (scale) and beta (shift) parameters to the normalized input
    result = gamma * normalized_x + beta

    return result

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):

    # Handle both batched and unbatched data
    if Q.ndim == 2:
        Q = Q[np.newaxis, ...]
        K = K[np.newaxis, ...]
        V = V[np.newaxis, ...]
        unbatched = True
    else:
        unbatched = False

    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Linear projection of Q, K and V
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)
  
    # Splitting the projection into heads
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
  
    # Scaled dot-product self-attention
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2))
    scaled_scores = scores / np.sqrt(d_k)
    attention_weights = softmax(scaled_scores, axis = -1)
    attention_out = np.matmul(attention_weights, V_heads)
  
    # Concatenate the heads
    attention_out = attention_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Projection of output
    result = np.matmul(attention_out, W_o)

    if unbatched:
        result = result[0]

    return result

def feed_forward_network(x, W1, b1, W2, b2):

    # First linear layer
    hidden = np.matmul(x, W1) + b1
  
    # GELU activation (non-linearity)
    activation = gelu(hidden)

    # Second linear layer
    result = np.matmul(activation, W2) + b2

    return result

def vit_encoder_block(x, embed_dim, num_heads, ffn_ratio = 4.0):

    # Handle both batched and unbatched data
    if x.ndim == 2:
        x = x[np.newaxis, ...]
        unbatched = True
    else:
        unbatched = False
    
    d_model = embed_dim
    d_ff = int(embed_dim * ffn_ratio)

    # For reproducibility
    np.random.seed(42)

    # Initialize the Multi-head attention weights
    W_q = np.random.randn(d_model, d_model) * 0.02
    W_k = np.random.randn(d_model, d_model) * 0.02
    W_v = np.random.randn(d_model, d_model) * 0.02
    W_o = np.random.randn(d_model, d_model) * 0.02

    # Initialize the Feed-forward network weights
    W1 = np.random.randn(d_model, d_ff) * 0.02
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.02
    b2 = np.zeros(d_model)

    # Initialize the Layer norm parameters for sub-layer 1
    gamma1 = np.ones(d_model)
    beta1 = np.zeros(d_model)
    # Initialize the Layer norm parameters for sub-layer 2
    gamma2 = np.ones(d_model)
    beta2 = np.zeros(d_model)

    # Sub-layer 1: Multi-head attention with Pre-LayerNorm
    x_norm1 = layer_norm(x, gamma1, beta1)
    attn_output = multi_head_attention(x_norm1, x_norm1, x_norm1, W_q, W_k, W_v, W_o, num_heads)
    # Residual connection 1
    x = x + attn_output  

    # Sub-layer 2: Feed-forward network with Pre-LayerNorm
    x_norm2 = layer_norm(x, gamma2, beta2)
    ffn_output = feed_forward_network(x_norm2, W1, b1, W2, b2)
    # Residual connection 2
    x = x + ffn_output 

    if unbatched:
        x = x[0]
    
    return x
