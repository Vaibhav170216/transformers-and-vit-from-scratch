import numpy as np

def gelu(x):
    """GELU activation function."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x, axis=-1):
    """Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-6):
    """Apply layer normalization."""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    output = gamma * x_normalized + beta
    return output

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    """Multi-head self-attention."""
    if Q.ndim == 2:
        Q = Q[np.newaxis, ...]
        K = K[np.newaxis, ...]
        V = V[np.newaxis, ...]
        unbatched = True
    else:
        unbatched = False
    
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    Q_proj = np.matmul(Q, W_q)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)

    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2))
    scaled_scores = scores / np.sqrt(d_k)
    attention_weights = softmax(scaled_scores, axis=-1)
    attention_output = np.matmul(attention_weights, V_heads)
    
    attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    output = np.matmul(attention_output, W_o)
    
    if unbatched:
        output = output[0]
    
    return output

def mlp(x, W1, b1, W2, b2):
    """MLP with GELU activation."""
    hidden = np.matmul(x, W1) + b1
    hidden = gelu(hidden)
    output = np.matmul(hidden, W2) + b2
    return output

class VisionTransformer:
    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 num_classes: int = 1000, embed_dim: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):
        """
        Initialize Vision Transformer. If weight arrays are provided, use them;
        otherwise initialize randomly.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes

        self.patch_dim = patch_size * patch_size * 3
        self.d_ff = int(embed_dim * mlp_ratio)
        self.seq_len = self.num_patches + 1

        np.random.seed(42)

        if W_patch is None:
            self.W_patch = np.random.randn(self.patch_dim, embed_dim) * 0.02
        else:
            self.W_patch = W_patch

        if cls_token is None:
            self.cls_token = np.random.randn(1, 1, embed_dim) * 0.02
        else:
            self.cls_token = cls_token

        if pos_embed is None:
            self.pos_embed = np.random.randn(self.seq_len, embed_dim) * 0.02
        else:
            self.pos_embed = pos_embed

        if encoder_weights is None:
            self.encoder_weights = []
            for _ in range(depth):
                layer_weights = {
                    'W_q': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'W_k': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'W_v': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'W_o': np.random.randn(embed_dim, embed_dim) * 0.02,
                    'W1': np.random.randn(embed_dim, self.d_ff) * 0.02,
                    'b1': np.zeros(self.d_ff),
                    'W2': np.random.randn(self.d_ff, embed_dim) * 0.02,
                    'b2': np.zeros(embed_dim),
                    'gamma1': np.ones(embed_dim),
                    'beta1': np.zeros(embed_dim),
                    'gamma2': np.ones(embed_dim),
                    'beta2': np.zeros(embed_dim),
                }
                self.encoder_weights.append(layer_weights)
        else:
            self.encoder_weights = encoder_weights

        if W_head is None:
            self.W_head = np.random.randn(embed_dim, num_classes) * 0.02
        else:
            self.W_head = W_head

    def patch_embed(self, images):

        if images.ndim == 3:
            images = images[np.newaxis, ...]

        batch_size, H, W, C = images.shape
        P = self.patch_size

        num_patches_h = H  // P
        num_patches_w = W  // P
        N = num_patches_h * num_patches_w

        patches = images.reshape(batch_size, num_patches_h, P, num_patches_w, P, C)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(batch_size, N, P * P * C)

        embeddings = np.matmul(patches, self.W_patch)

        return embeddings

    def add_cls_and_pos_embed(self, patch_embeddings):

        batch_size = patch_embeddings.shape[0]

        cls_tokens = np.tile(self.cls_token, (batch_size, 1, 1))
        tokens = np.concatenate([cls_tokens, patch_embeddings], axis = 1)

        tokens = tokens + self.pos_embed

        return tokens

    def encoder_block(self, x, layer_weights):

        norm1 = layer_norm(x, layer_weights['gamma1'], layer_weights['beta1'])
        attn_output = multi_head_attention(
            norm1, norm1, norm1,
            layer_weights['W_q'],
            layer_weights['W_k'],
            layer_weights['W_v'],
            layer_weights['W_o'],
            self.num_heads
        )
        x = x + attn_output

        norm2 = layer_norm(x, layer_weights['gamma2'], layer_weights['beta2'])
        mlp_output = mlp(
            norm2,
            layer_weights['W1'],
            layer_weights['b1'],
            layer_weights['W2'],
            layer_weights['b2']
        )
        x = x + mlp_output

        return x

    def classification_head(self, encoder_output):

        h_cls = encoder_output[:, 0, :]

        eps = 1e-6
        mean = np.mean(h_cls, axis = -1, keepdims = True)
        variance = np.var(h_cls, axis = -1, keepdims = True)
        normalized_h = (h_cls - mean) / np.sqrt(variance + eps)

        logits = np.matmul(normalized_h, self.W_head)

        return logits

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        """

        single_image = (x.ndim == 3)
        if single_image:
            x = x[np.newaxis, ...]

        patch_embeddings = self.patch_embed(x)

        out = self.add_cls_and_pos_embed(patch_embeddings)

        for layer_idx in range(self.depth):
            out = self.encoder_block(out, self.encoder_weights[layer_idx])

        logits = self.classification_head(out)

        if single_image:
            logits = logits[0]

        return logits
