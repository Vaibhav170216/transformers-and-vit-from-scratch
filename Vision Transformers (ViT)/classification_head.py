import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    """
    Classification head for ViT.
    W_head: projection matrix (D, num_classes). If None, initialize randomly.
    """

    # Handling both single and batched inputs
    if encoder_output.ndim == 2:
        encoder_output = encoder_output[np.newaxis, ...]
        unbatched = True
    else:
        unbatched = False

    batch_size, seq_len, embed_dim = encoder_output.shape

    # Extract cls token which is at position 0 in the sequence
    cls_token = encoder_output[:, 0, :]

    # Normalize the token by calculating mean and variance (LayerNorm)
    eps = 1e-6
    mean = np.mean(cls_token, axis = -1, keepdims = True)
    variance = np.var(cls_token, axis = -1, keepdims = True)
    normalized_head = (cls_token - mean) / np.sqrt(variance + eps)

    # Linear projection by randomly initializing weight projection matrix 
    if W_head is None:
        np.random.seed(42)
        W_head = np.random.randn(embed_dim, num_classes) * 0.02

    # Calculate class raw logits
    logits = np.matmul(normalized_head, W_head)

    if unbatched:
        logits = logits[0]

    return logits

