'''
Implementation of the configuration management system for the digital-comms project.
This module provides functionality to load, manage, and access configuration settings
'''
from dataclasses import dataclass

# Define a dataclass to hold configuration settings
@dataclass(frozen=True, slots=True)
class SimulationConfig:
  '''
  Configuration settings for the simulation environment.
  '''
  ebn0_db: float = 6.0 # Energy per bit to noise power spectral density ratio in dB
  number_of_bits: int = 100_000 # Total number of bits to simulate
  random_seed: int = 42 # Seed for random number generation to ensure reproducibility

  # Define a method to display the configuration settings
  def __post_init__(self) -> None:
    if self.number_of_bits <= 0:
      raise ValueError("Number of bits must be a positive integer.")
    
    if self.number_of_bits % 2 != 0:
      raise ValueError("Number of bits must be an even integer for proper modulation.")