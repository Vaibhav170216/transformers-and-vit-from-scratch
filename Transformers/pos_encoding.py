import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    # Initialize the position embedding matrix
    pe = np.zeros((seq_length, d_model))

    # Create position indices
    pos = np.arange(seq_length).reshape(-1, 1)

    # Create dimension indices
    dim = np.arange(d_model // 2)

    # Calculate the division term
    division_term = np.power(10000.0, 2 * dim / d_model)

    # Calculate scaled positions or angles
    scaled_pos = pos / division_term

    # Apply sin to even indices
    pe[:, 0::2] = np.sin(scaled_pos)

    # Apply cos to the odd indices
    pe[:, 1::2] = np.cos(scaled_pos)

    return pe
    
