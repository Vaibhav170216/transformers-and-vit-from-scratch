import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """

    # First linear transformation where d_model (model dimension) changes to d_ff (feed forward network dimension) [d_ff = 4 x d_model]
    hidden = np.matmul(x, W1) + b1

    # ReLU activation to introduce non-linearity               
    relu_act = np.maximum(0, hidden)

    # Second linear transformation where d_ff changes back to d_model               
    result = np.matmul(relu_act, W2) + b2

    return result
