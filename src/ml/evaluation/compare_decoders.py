"""
Compare Soft Viterbi and neural decoding over coded QPSK-AWGN.
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import sys

from src.core.channel.awgn import AWGNChannel
from src.core.coding.convolutional import ConvolutionalEncoder
from src.core.coding.soft_viterbi import SoftViterbiDecoder
from src.core.modulation.qpsk import QPSK
from src.ml.models.service import NeuralDecoderService
from src.exception.exception import CustomException

# Define function to run the comparison between Soft Viterbi and Neural Decoding
def main() -> None:
  try:
    # Initialize the encoder, decoders, and modulation scheme
    encoder = ConvolutionalEncoder()
    viterbi_decoder = SoftViterbiDecoder()
    qpsk = QPSK()
    neural_decoder = NeuralDecoderService("results/models/best_decoder.keras")

    # Define random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(123)
    ebn0_values_db = np.arange(-2.0, 9.0, 1.0)
    frames_per_snr = 100
    information_bits_per_frame = 1_006

    # Initialize lists to store BER and latency results for both decoders
    viterbi_ber: list[float] = []
    neural_ber: list[float] = []
    viterbi_latency_ms: list[float] = []
    neural_latency_ms: list[float] = []

    # Loop over each Eb/N0 value in the defined range
    for ebn0_db in ebn0_values_db:
      # Initialize counters for errors, total bits, and decoding times
      viterbi_errors = 0
      neural_errors = 0
      total_bits = 0
      total_viterbi_time = 0.0
      total_neural_time = 0.0

      for _ in range(frames_per_snr):
        # Generate random information bits for the current frame
        information_bits = rng.integers(low=0,high=2,size=information_bits_per_frame,dtype=np.uint8)
        encoded_bits = encoder.encode(information_bits, terminate=True)

        # Transmit the encoded bits over an AWGN channel using QPSK modulation
        transmitted_symbols = qpsk.map(encoded_bits)
        channel = AWGNChannel(ebn0_db=float(ebn0_db), bits_per_symbol=2, symbol_energy=1.0, code_rate=0.5)
        received_symbols = channel.transmit(transmitted_symbols, rng)

        # Soft QPSK demapping of received symbols to LLRs
        llrs = qpsk.soft_demap(symbols=received_symbols, noise_variance=channel.noise_variance)

        # Decode the received LLRs using both Soft Viterbi and Neural Decoders
        viterbi_start = perf_counter()
        viterbi_bits = (
          viterbi_decoder.decode(
            llrs,
            terminated=True
            )
          )
        total_viterbi_time += (perf_counter() - viterbi_start)

        # Decode using the neural decoder and measure the time taken
        neural_start = perf_counter()
        neural_bits = neural_decoder.decode(
          llrs,
          terminated=True,
        )
        total_neural_time += (perf_counter() - neural_start)

        # Calculate the number of bit errors for both decoders and accumulate the total bits processed
        viterbi_errors += int(
          np.count_nonzero(
            information_bits
            != viterbi_bits
          )
        )
        neural_errors += int(
          np.count_nonzero(
            information_bits
            != neural_bits
          )
          )

        # Accumulate the total number of information bits processed for BER calculation
        total_bits += (information_bits.size)

      # Calculate the current BER for both Soft Viterbi and Neural Decoders based on the accumulated errors and total bits processed
      current_viterbi_ber = (viterbi_errors / total_bits)
      current_neural_ber = (neural_errors / total_bits)

      # Append the current BER and average latency results for both decoders to their respective lists for later plotting and analysis
      viterbi_ber.append(current_viterbi_ber)
      neural_ber.append(current_neural_ber)
      viterbi_latency_ms.append(1_000.0 * total_viterbi_time / frames_per_snr)
      neural_latency_ms.append(1_000.0 * total_neural_time / frames_per_snr)

      # Print the results for the current Eb/N0 value, including BER and average latency for both decoders
      print(f"Eb/N0={ebn0_db:5.1f} dB | Viterbi={current_viterbi_ber:.6e} | Neural={current_neural_ber:.6e} | Viterbi latency={viterbi_latency_ms[-1]:.3f} ms | Neural latency={neural_latency_ms[-1]:.3f} ms"
      )

    # Define the output directory for saving the BER and latency plots
    output_directory = Path("results/phase_5")
    output_directory.mkdir(parents=True, exist_ok=True,)

    # Calculate the minimum measurable BER based on the number of frames and 
    # information bits per frame to avoid plotting zero values on a logarithmic scale
    minimum_measurable_ber = (1.0 / (frames_per_snr * information_bits_per_frame))

    # Prepare the BER data for plotting by ensuring that the values are above the 
    # minimum measurable BER to avoid issues with logarithmic scaling in the plots
    viterbi_plot = np.maximum(np.asarray(viterbi_ber), minimum_measurable_ber)
    neural_plot = np.maximum(np.asarray(neural_ber), minimum_measurable_ber)

    # Plot the BER comparison between Soft Viterbi and Neural Decoders using a semi-logarithmic scale
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebn0_values_db, viterbi_plot, marker="o", label="Soft Viterbi")
    plt.semilogy(ebn0_values_db, neural_plot, marker="s", label="Neural decoder")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Information-bit BER")
    plt.title("Soft Viterbi vs Neural Decoder")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # Define the path for saving the BER comparison plot and save it to the output directory
    ber_path = (output_directory / "phase5_decoder_ber_comparison.png")
    plt.savefig(ber_path, dpi=300)
    plt.show()

    # Plot the average decoding latency comparison between Soft Viterbi and Neural Decoders
    plt.figure(figsize=(10, 6))
    plt.plot(ebn0_values_db,viterbi_latency_ms,marker="o",label="Soft Viterbi")
    plt.plot(ebn0_values_db,neural_latency_ms,marker="s",label="Neural decoder")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Average decoding latency (ms/frame)")
    plt.title("Decoder Latency Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Define the path for saving the latency comparison plot and save it to the output directory
    latency_path = (output_directory / "phase5_decoder_latency_comparison.png")
    plt.savefig(latency_path, dpi=300)
    plt.show()

    # Print the paths where the BER and latency plots have been saved
    print(f"BER plot saved to: {ber_path}")
    print(f"Latency plot saved to: {latency_path}")
  except CustomException as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()