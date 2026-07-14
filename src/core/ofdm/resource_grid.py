'''
LTE-inspired OFDM resource grid configuration class.
This class defines the configuration parameters for an OFDM resource grid based on LTE specifications.
'''
import numpy as np
from numpy.typing import NDArray
from src.core.ofdm.config import OFDMConfig

# Define bit array and complex array types for better type hinting
ComplexArray = NDArray[np.complex128]
IndexArray = NDArray[np.int64]

# Define resource grid class
class ResourceGridMapper:
  def __init__(self, config: OFDMConfig):
    self._config = config
  
  # Define property to calculate the number of active indices
  def active_indices(self) -> IndexArray:
    '''
    Return active FFT-bin indices in fftshifted ordering.
    For NFFT=128 and 72 active subcarriers:
    - 36 bins below DC
    - DC bin unused
    - 36 bins above DC
    '''
    # Define center and half of active subcarriers
    center = self._config.fft_size // 2
    half = self._config.active_subcarriers // 2

    # Define negative frequencies
    negative_freqs = np.arange(center - half, center, dtype=np.int64)

    # Define positive frequencies
    positive_freqs = np.arange(center + 1, center + half + 1, dtype=np.int64)

    # Concatenate negative and positive frequencies
    return np.concatenate(
      [negative_freqs, positive_freqs]
    )
  
  # Define method to create empty grid
  def create_empty_grid(self) -> ComplexArray:
    '''Create an empty grid in fftshifted frequency order'''
    return np.zeros(
      (
        self._config.symbols_per_subframe, 
        self._config.fft_size
      ),
      dtype=np.complex128,
    )
  
  # Define method to map symbols to grid
  def map_symbols(self, symbols: ComplexArray) -> ComplexArray:
    '''Map serialized QPSK symbols into active grid bins'''
    # Define symbols 
    symbols = np.asarray(symbols, dtype=np.complex128)
    
    # Check if the symbol dimension is correct
    if symbols.ndim != 1:
      raise ValueError("Symbols must be a 1D array.")
    
    # Define expected symbols
    expected_symbols = self._config.qpsk_symbols_per_subframe

    # Check if the symbols size is correct
    if symbols.size != expected_symbols:
      raise ValueError(
        f"Expected {expected_symbols} symbols, but got {symbols.size}."
      )
    
    # Define empty grid
    empty_grid = self.create_empty_grid()
    empty_grid[:, self.active_indices()] = symbols.reshape(
      self._config.symbols_per_subframe,
      self._config.active_subcarriers
    )
    return empty_grid
  
  # Define method to extract symbols from grid
  def extract_symbols(self, grid: ComplexArray) -> ComplexArray:
    '''Extract active subcarriers and serialize them'''
    # Define grid
    grid = np.asarray(grid, dtype=np.complex128)

    # Define expected grid shape
    expected_shape = (
      self._config.symbols_per_subframe,
      self._config.fft_size
    )

    # Check if the grid shape is correct
    if grid.shape != expected_shape:
      raise ValueError(
        f"Expected grid shape {expected_shape}, but got {grid.shape}."
      )
    
    # Return serialized symbols from active indices
    return grid[:, self.active_indices()].reshape(-1)