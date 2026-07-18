'''
Conversion between UTF-8 text, bytes, and binary arrays.
'''
import numpy as np
from numpy.typing import NDArray

# Define type alias for binary array
BitArray = NDArray[np.uint8]

# Define class TextCodec for encoding and decoding text
class TextCodec:
  encoding: str = "utf-8"

  # Define class method to encode text to binary array
  @classmethod
  def encode(cls, text: str) -> BitArray:
    """
    Convert a Python string into UTF-8 bits.
    Example:
      "A" → b"\\x41" → [0, 1, 0, 0, 0, 0, 0, 1]
    """
    # Check if the input text is a string
    if not isinstance(text, str):
      raise ValueError("Input text must be a string.")
    
    # Check if the input text is empty
    if text == "":
      raise ValueError("Input text must not be empty.")
    
    # Encode the text to bytes using UTF-8 encoding
    payload_bytes = text.encode(cls.encoding)

    # Convert the bytes to a binary array
    bytes_array = np.frombuffer(payload_bytes, dtype=np.uint8)

    # Convert the binary array to a bit array
    return np.unpackbits(bytes_array, bitorder="big").astype(np.uint8)
  
  # Define class method to decode binary array to text
  @classmethod
  def decode(cls, bits: BitArray, errors: str = "replace") -> str:
    """
    Reconstruct UTF-8 text from bits.
    """
    # Validate the input bits
    bits = cls._validate_bits(bits)

    # Check if the length of the bits is a multiple of 8
    if bits.size % 8 != 0:
      raise ValueError("Input bits length must be a multiple of 8.")
    
    # Define byte array
    byte_array = np.packbits(bits, bitorder="big")

    # Convert the byte array to bytes
    return byte_array.tobytes().decode(cls.encoding, errors=errors)

  
  # Define static method to validate bits
  @staticmethod
  def _validate_bits(bits: BitArray) -> BitArray:
    # Define bits as a numpy array of type uint8
    bits = np.asarray(bits, dtype=np.uint8)

    # Check if the input bits are a one-dimensional array
    if bits.ndim != 1:
      raise ValueError("Input bits must be a 1D array.")
    
    # Check if the input bits are empty
    if bits.size == 0:
      raise ValueError("Input bits must not be empty.")
    
    # Check if the bits value are not binary (0 or 1)
    if not np.all((bits == 0) | (bits == 1)):
      raise ValueError("Input bits must contain only binary values (0 or 1).")\
  
    # Return the validated bits
    return bits