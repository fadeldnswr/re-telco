"""
Inference wrapper for the trained neural decoder.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

# Define type aliases for arrays
FloatArray = NDArray[np.float64]
BitArray = NDArray[np.uint8]

# Define NeuralDecoder class to wrap the trained neural decoder model
class NeuralDecoderService:
  encoder_memory: int = 2
  def __init__(self, model_path: str | Path) -> None:
    # Load the trained neural decoder model from the specified path
    model_path = Path(model_path)

    # Check if the model path exists and is a directory
    if not model_path.exists():
      raise FileNotFoundError(f"Model path '{model_path}' does not exist or is not a directory.")
    
    # Load the trained model using TensorFlow's load_model function
    self._model = tf.keras.models.load_model(model_path)
  
  # Define method to decode
  def decode(self, coded_llrs: FloatArray, terminated: bool = True) -> BitArray:
    """
    Decode one convolutionally coded LLR sequence.
    Args:
      coded_llrs:
        One-dimensional sequence:
        [LLR0, LLR1, LLR2, LLR3, ...]

    Returns:
      Predicted information bits.
    """
    # Define coded llrs
    coded_llrs = np.asarray(coded_llrs, dtype=np.float32)

    # Check its dimension and size
    if coded_llrs.ndim != 1:
      raise ValueError("Received LLRs must be a 1D array.")
    if coded_llrs.size == 0:
      raise ValueError("Received LLRs must not be empty.")
    if coded_llrs.size % 2 != 0:
      raise ValueError("Received LLRs size must be a multiple of 2.")
    
    # Define trellis steps
    trellis_steps = coded_llrs.size // 2

    # Define model input
    model_input = coded_llrs.reshape(1, trellis_steps, 2)

    # Use the trained model to predict the information bits
    logits = self._model(model_input, training=False).numpy()[0, :, 0]

    # Convert logits to binary bits using a threshold of 0
    decoded_with_tail = (logits >= 0.0).astype(np.uint8)

    # Check if the decoder is terminated and remove tail bits if necessary
    if terminated:
      if (decoded_with_tail.size < self.encoder_memory):
        raise RuntimeError("Decoded bits size is smaller than encoder memory, cannot remove tail bits.")
      return decoded_with_tail[:-self.encoder_memory]
    
    # Return the decoded bits without removing tail bits
    return decoded_with_tail