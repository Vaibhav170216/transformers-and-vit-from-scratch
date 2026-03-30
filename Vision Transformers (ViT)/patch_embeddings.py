import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    """

    # For reproducibility
    np.random.seed(42)
    
    B, H, W, C = image.shape  # Image dimensions (Batch x Height x Width x Channels)
    P = patch_size
    patch_dim = P * P * C

    # Verify patch size divides Height and Width evenly
    assert H % P == 0
    assert W % P == 0

    # Calculate number of patches
    num_patches_h = H // P
    num_patches_w = W // P

    # Total number of patches
    N = num_patches_h * num_patches_w
  
    # Reshape image to extract patches
    patches = image.reshape(B, num_patches_h, P, num_patches_w, P, C)

    # Reorder dimensions to group the patches together
    patches = patches.transpose(0, 1, 3, 2, 4, 5)

    # Reshape to get sequence of patches
    patches = patches.reshape(B, N, P, P, C)

    # Flatten each patch
    flat_patches = patches.reshape(B, N, patch_dim)

    # Initialize the projection matrix
    W_proj = np.random.randn(patch_dim, embed_dim) * np.sqrt(2.0 / patch_dim)

    # Apply Linear projection: (B, N, P^2*C) @ (P^2*C, D) --> (B, N, D)
    embeddings = np.matmul(flat_patches, W_proj)

    return embeddings
