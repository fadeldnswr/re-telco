'''
Reusable channel profiles for different use cases.
'''
import numpy as np
from src.core.channel.static_multipath import StaticMultipathChannel

# Create function to define three path channel profile
def create_three_path_channel() -> StaticMultipathChannel:
  '''
  Static channel with delays shorter than the minimum LTE-like CP.
    Delays: 0, 2, and 5 samples
    Powers: 0, -3, and -8 dB
  '''
  # Define delays in samples
  delays = np.array([0, 2, 5], dtype=np.int64)
  
  # Define powers in dB and convert to linear scale
  powers_db = np.array([0.0, -3.0, -8.0], dtype=np.float64)

  # Define phase in radian
  phase_rad = np.array([0.0, 0.4, -0.8], dtype=np.float64)

  # Define amplitudes derived from powers
  amplitudes = 10.0 ** (
    powers_db / 20.0
  )

  # Define gains derived from amplitude and phase
  gains = amplitudes * np.exp(
    1j * phase_rad
  )

  # Return static multipath channel with defined delays and gains
  return StaticMultipathChannel(
    delays=delays, gains=gains.astype(np.complex128),
    normalize_power=True
  )