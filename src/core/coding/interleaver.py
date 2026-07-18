"""
Deterministic block interleaver.

The same permutation is used by the transmitter and receiver.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# Define index array
IndexArray = NDArray[np.int64]

# Define a class for the interleaver
@dataclass(frozen=True, slots=True)
class BlockInterleaver:
  """
    Deterministic permutation-based block interleaver.

    It can process bits, floating-point LLRs, or other one-dimensional
    arrays as long as their lengths match the permutation length.
  """
  permutation: IndexArray
  inverse_permutation: IndexArray

  # Define class method to create an interleaver from a given permutation
  @classmethod
  def create(cls, length: int, random_seed: int | None = None) -> "BlockInterleaver":
    # Check if length is a positive integer
    if length <= 0:
      raise ValueError("Length must be a positive integer.")
    
    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed=random_seed)

    # Generate a random permutation of indices
    permutation = rng.permutation(length).astype(np.int64)
    inverse_permutation = np.argsort(permutation).astype(np.int64)

    # Return a new instance of BlockInterleaver with the generated permutation
    return cls(
      permutation=permutation,
      inverse_permutation=inverse_permutation
    )
  
  # Define property to get the length of the interleaver
  @property
  def length(self) -> int:
    return int(self.permutation.size)
  
  # Define function to validate length
  def _validate_length(self, values: np.ndarray) -> None:
    # Check if the input values are a one-dimensional array
    if values.ndim != 1:
      raise ValueError("Input values must be a 1D array.")
    
    # Check if the size of the input values matches the interleaver length
    if values.size != self.length:  
      raise ValueError(
        f"Input values size {values.size} does not match interleaver length {self.length}."
      )
  
  # Define method to interleave an input array using the permutation
  def interleave(self, values: np.ndarray) -> np.ndarray:
    # Define values as a numpy array
    values = np.asarray(values)

    # Validate the length of the input values
    self._validate_length(values)

    # Return the interleaved values using the permutation
    return values[self.permutation]
  
  # Define method to deinterleave an input array using the inverse permutation
  def deinterleave(self, values: np.ndarray) -> np.ndarray:
    '''
    Restore the original order of the input values using the inverse permutation.
    '''
    # Define values as a numpy array
    values = np.asarray(values)

    # Validate the length of the input values
    self._validate_length(values)

    # Return the deinterleaved values using the inverse permutation
    return values[self.inverse_permutation]