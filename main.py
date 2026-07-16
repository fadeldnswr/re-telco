'''
Main application entry point for the digital-comms project.
This script initializes the logging system, sets up the application environment,
'''
from src.core.system.phase2.phase2_ofdm_loopback import plot_and_run
from src.core.system.phase2.phase2_ofdm_loopback import Phase2Simulator
from src.logging.logging import logger
from src.exception.exception import CustomException

import sys
import os

# Define main function to run the application
if __name__ == "__main__":
  try:
    # Run the Phase 2 OFDM loopback simulation and visualization
    simulator = Phase2Simulator()
    single_result = simulator.run(ebn0_db=6.0, random_seed=42)

    # Run a batch of simulations to compute the BER over multiple subframes
    batch_result = (simulator.run_ber_experiment(
      ebn0_db=6.0,
      number_of_subframes=500,
      random_seed=42,
      )
    )

    # Print the results of the single-frame simulation and batch BER experiment
    print("Phase 2B: QPSK-OFDM over AWGN")
    print("-" * 48)
    print(f"FFT size                : {simulator.config.fft_size}")
    print(f"Active subcarriers      : {simulator.config.active_subcarriers}")
    print(f"Waveform length         : {single_result.waveform_length}")
    print(f"Waveform bit energy     : {single_result.waveform_bit_energy:.6e}")
    print(f"Noise variance          : {single_result.noise}")
    print(f"Single-frame errors     : {single_result.number_of_bit_errors}")
    print(f"Single-frame BER        : {single_result.ber:.6e}")
    print(f"Frequency-domain EVM    : {single_result.evm_rms:.6f}")
    print()
    print("Batch BER experiment")
    print(f"Subframes               : {batch_result.total_subframes}")
    print(f"Total bits              : {batch_result.total_bits}")
    print(f"Total errors            : {batch_result.total_bit_errors}")
    print(f"BER                     : {batch_result.ber:.6e}")
  except Exception as e:
    raise CustomException(e, sys)