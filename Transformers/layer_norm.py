import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """

    # Compute mean across the last dimension or feature dimension
    mean = np.mean(x, axis = -1, keepdims = True)

    # Comput variance across the last dimension
    variance = np.var(x, axis = -1, keepdims = True)

    # Normalization to zero mean and unit variance
    x_norm = (x - mean) / np.sqrt(variance + eps)

    # Apply gamma (scale) and beta (shift) which are learnable parameters to the normalized input
    result = gamma * x_norm + beta

    return result
