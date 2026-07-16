'''
Run Phase 3A simulation for static multipath channel with and without equalization.
'''

import sys

from src.core.system.phase3.phase3_simulator import EqualizerType, Phase3Simulator
from src.exception.exception import CustomException
from src.core.system.phase3.phase3_ber_curve import run_ber_curve

# Define main function to run the Phase 3A simulation
def main() -> None:
  try:
    # Initialize the Phase 3 simulator
    simulator = Phase3Simulator()

    # Run simulation without equalization and with perfect-CSI ZF equalization
    no_equalization = simulator.run(
      ebn0_db=10.0,
      equalizer_type=EqualizerType.NONE,
      random_seed=42,
    )
    # Run simulation with perfect-CSI ZF equalization
    zf_result = simulator.run(
      ebn0_db=10.0,
      equalizer_type=EqualizerType.ZF,
      random_seed=42,
    )

    # Run simulation with MMSE equalization
    mmse_result = simulator.run(
      ebn0_db=10.0,
      equalizer_type=EqualizerType.MMSE,
      random_seed=42,
    )

    print("Phase 3B: Static Multipath with ZF and MMSE Equalization")
    print("-" * 48)

    # Print simulation results for both cases
    print(f"Maximum channel delay   : {zf_result.maximum_channel_delay}")
    print(f"Minimum CP length       : {zf_result.minimum_cp_length}")
    print(f"CP condition satisfied  : {zf_result.cp_condition_satisfied}")
    print()
    print("Without equalization")
    print(f"BER                     : {no_equalization.ber:.6e}")
    print(f"EVM                     : {no_equalization.unequalized_evm:.6f}")
    print()
    print("Perfect-CSI ZF")
    print(f"BER                     : {zf_result.ber:.6e}")
    print(f"EVM                     : {zf_result.equalized_evm:.6e}")
    print()
    print("MMSE")
    print(f"BER                     : {mmse_result.ber:.6e}")
    print(f"EVM                     : {mmse_result.equalized_evm:.6e}")
  except Exception as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()
  run_ber_curve()