'''
Configuration file for OFDM module.
This module provides configuration parameters for the OFDM system.
'''
from dataclasses import dataclass

# Define a dataclass for OFDM configuration parameters
@dataclass(frozen=True, slots=True)
class OFDMConfig:
  '''
  LTE-inspired 1.4 MHz OFDM configuration parameters.
  '''
  fft_size: int = 128 # FFT size
  resource_blocks: int = 6 # Number of resource blocks
  subcarriers_per_rb: int = 12 # Number of subcarriers per resource block
  symbols_per_slot: int = 7 # Number of symbols per slot
  slots_per_subframe: int = 2 # Number of slots per subframe
  subcarrier_spacing_hz: float = 15_000.0 # Subcarrier spacing in Hz
  long_cp_samples: int = 10 # Number of samples in long cyclic prefix
  short_cp_samples: int = 9 # Number of samples in short cyclic prefix

  # Define post init method to calculate derived parameters
  def __post_init__(self) -> None:
    # Check if the fft size is negative
    if self.fft_size <= 0:
      raise ValueError("FFT size must be a positive integer.")
    
    # Check if the resource blocks is negative
    if self.resource_blocks <= 0:
      raise ValueError("Number of resource blocks must be a positive integer.")
    
    # Check if the active subcarrier is more than the fft size
    if self.active_subcarriers > self.fft_size:
      raise ValueError("Number of active subcarriers cannot exceed FFT size.")
    
    # Check if the active subcarrier is not even
    if self.active_subcarriers % 2 != 0:
      raise ValueError("Number of active subcarriers must be an even number for symmetric mapping.")
    
    # Check if long cyclic prefix samples is more than the fft size
    if self.long_cp_samples >= self.fft_size:
      raise ValueError("Number of long cyclic prefix samples must be less than FFT size.")
    
    # Check if short cyclic prefix samples is more than the fft size
    if self.short_cp_samples >= self.fft_size:
      raise ValueError("Number of short cyclic prefix samples must be less than FFT size.")
  
  # Define property to calculate the number of active subcarriers
  @property
  def active_subcarriers(self) -> int:
    return self.resource_blocks * self.subcarriers_per_rb
  
  # Define property to calculate the number of samples per symbol
  @property
  def symbols_per_subframe(self) -> int:
    return self.symbols_per_slot * self.slots_per_subframe
  
  # Define property to calculate sample rate
  @property
  def sample_rate_hz(self) -> float:
    return self.subcarrier_spacing_hz * self.fft_size
  
  # Define property to calculate qpsk symbols per subframe
  @property
  def qpsk_symbols_per_subframe(self) -> int:
    return self.active_subcarriers * self.symbols_per_subframe
  
  # Define property to calculate qpsk bits per subframe
  @property
  def qpsk_bits_per_subframe(self) -> int:
    return 2 * self.qpsk_symbols_per_subframe
  
  # Define method to calculate the cyclic prefix length
  def cyclic_prefix_length(self, symbol_index: int) -> int:
    '''Return LTE-scaled normal cyclic prefix length for a given symbol index.'''
    # Check if symbol index is valid
    if not 0 <= symbol_index < self.symbols_per_subframe:
      raise ValueError(f"Symbol index must be in the range [0, {self.symbols_per_subframe - 1}].")
    
    # Define symbols in slot
    symbol_in_slot = symbol_index & self.symbols_per_slot
    if symbol_in_slot == 0: # First symbol in slot has long cyclic prefix
      return self.long_cp_samples
    
    # Return short cyclic prefix samples for other symbols
    return self.short_cp_samples
  
  # Define property to calculate samples per slot
  @property
  def samples_per_slot(self) -> int:
    return (
      self.fft_size + self.long_cp_samples 
      + (self.symbols_per_slot - 1)
      * (self.fft_size + self.short_cp_samples)
    )
  
  # Define property to calculate samples per subframe
  @property
  def samples_per_subframe(self) -> int:
    return self.samples_per_slot * self.slots_per_subframe

# Define OFDM configuration instance with default parameters
config = OFDMConfig()

# Define assertions to validate the configuration parameters
assert config.active_subcarriers == 72
assert config.symbols_per_subframe == 14
assert config.qpsk_symbols_per_subframe == 1008
assert config.qpsk_bits_per_subframe == 2016
assert config.sample_rate_hz == 1_920_000
assert config.samples_per_slot == 960
assert config.samples_per_subframe == 1920