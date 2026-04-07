import numpy as np

def class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """

    # Handle both batch and non-batch (single) inputs
    if patches.ndim == 2:
        # Add batch_dimension
        patches = patches[np.newaxis, ...]
        single_input = True
    else:
        single_input = False

    batch_size = patches.shape[0]

    # For reproducibility
    np.random.seed(42)

    # Random initialization of learnable [CLS] token 
    cls_token = np.random.randn(1, 1, embed_dim) * 0.02

    # Broadcast the [CLS] token to match the batach size
    batch_cls_tokens = np.tile(cls_token, (batch_size, 1, 1))

    # Concatenate the [CLS] token at the beginning of the sequence
    result = np.concatenate([batch_cls_tokens, patches], axis = 1)

    # If 2D input then remove batch dimension
    if single_input:
        result = result[0]

    return result
