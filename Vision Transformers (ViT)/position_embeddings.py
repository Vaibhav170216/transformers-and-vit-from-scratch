import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    patches: Patch embeddings (N, D) where N = num_patches and D = embedding_dimension
    """

    # For reproducibility
    np.random.seed(42) 
  
    # Random initialization for learnable position embeddings during training
    pos_embeddings = np.random.randn(num_patches, embed_dim) * 0.02

    # Add position embeddings to patches
    result = patches + pos_embeddings
    
    return result
