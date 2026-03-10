import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Formula = Attention(Q, K, V) = softmax(Q.K^T/ sqrt(d_k)).V
    """

    # key dimension
    d_k = Q.size(-1)

    # Computing atention scores 
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Scaling the scores for stability so that gradients remain healthy and do not vanish
    scaled_scores = scores/math.sqrt(d_k)

    # Apply softmax across the key dimension to get normalized attention weights
    attention_weights = F.softmax(scaled_scores, dim = -1)

    # Final output resulting in wieghted average of all Value vectors
    result = torch.matmul(attention_weights, V)

    return result
