'''
Monte carlo simulation using different AWGN
'''
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy.typing import NDArray
from src.exception.exception import CustomException
from src.ml.evaluation.text_decoder_comparison import TextDecoderComparison
from src.ml.evaluation.evaluation_models import (
  DecoderAggregateResult, DecoderTextResult, TextComparisonResult, 
  BitArray, FloatArray, TextMonteCarloResult, TextSNRResult
)

# Define class for running Monte Carlo trials for text decoder comparison
class TextMonteCarloEvaluator:
  def __init__(self, neural_model_path: str | Path) -> None:
    self._comparison = TextDecoderComparison(neural_model_path=neural_model_path)
  
  # Define static method to build decoder result
  @staticmethod
  def _build_decoder_result(
    total_bit_errors: int, total_bits: int, total_frame_errors: int, 
    total_frames: int, exact_text_success: int, total_trials: int,
    latencies_ms: list[float]) -> DecoderAggregateResult:
    # Check if total_bits is zero to avoid division by zero
    if total_bits <= 0:
      raise ValueError("Total bits must be greater than zero.")
    if total_frames <= 0:
      raise ValueError("Total frames must be greater than zero.")
    if total_trials <= 0:
      raise ValueError("Total trials must be greater than zero.")
    
    # Define latency array
    latency_array = np.asarray(latencies_ms, dtype=np.float64)

    # Check if the latency array size matches the total trials
    if latency_array.size != total_trials:
      raise ValueError("Latency array size must match total trials.")
    
    # Return result
    return DecoderAggregateResult(
      total_bit_errors=total_bit_errors,
      total_bits=total_bits,
      ber=total_bit_errors / total_bits,
      total_frame_errors=total_frame_errors,
      total_frames=total_frames,
      frame_error_rate=total_frame_errors / total_frames,
      exact_text_successes=exact_text_success,
      total_trials=total_trials,
      exact_text_success_rate=exact_text_success / total_trials,
      mean_latency_ms=float(np.mean(latency_array)),
      median_latency_ms=float(np.median(latency_array)),
      latency_p95_ms=float(np.percentile(latency_array, 95))
    )
  
  # Define method to evaluate the result
  def evaluate(
    self, text: str, ebn0_values_db: list[float] | FloatArray, 
    trials_per_snr: int = 100, base_random_seed: int = 42, 
    print_progress: bool = True) -> TextMonteCarloResult:
    # Check if there is no text
    if not text:
      raise ValueError("Input text must not be empty.")
    
    # Check if trials_per_snr is positive
    if trials_per_snr <= 0:
      raise ValueError("Trials per SNR must be a positive integer.")
    
    # Define ebn0 array
    ebn0_values_db = np.asarray(ebn0_values_db, dtype=np.float64)

    # Check if the dimension of ebn0_values_db is 1D
    if ebn0_values_db.ndim != 1:
      raise ValueError("Eb/N0 values must be a 1D array.")
    
    # Check if the size of ebn0_values_db is zero
    if ebn0_values_db.size == 0:
      raise ValueError("Eb/N0 values must not be empty.")
    
    # Define list to hold the results of the Monte Carlo trials
    snr_results: list[TextSNRResult] = []
    payload_bits_per_trial: int | None = None
    frames_per_trial: int | None = None

    # Iterate through the SNR values
    for snr_index, ebn0_db in enumerate(ebn0_values_db):
      # Define variables to hold the total bit errors, total bits, total frame errors,
      # total frames, exact text successes, and latencies for Viterbi and neural decoders
      viterbi_bit_errors = 0
      neural_bit_errors = 0
      viterbi_frame_errors = 0
      neural_frame_errors = 0
      viterbi_text_successes = 0
      neural_text_successes = 0
      total_bits = 0
      total_frames = 0

      # Define list to hold the latencies for Viterbi and neural decoders
      viterbi_latencies_ms: list[float] = []
      neural_latencies_ms: list[float] = []

      # Start the performance counter for the SNR evaluation
      snr_start = perf_counter()

      # Iterate through the number of trials for the current SNR value
      for trial_index in range(trials_per_snr):
        # Define random seed for the current trial
        random_seed = base_random_seed + snr_index * 100000 + trial_index

        # Transmit the text and get the comparison result
        result = self._comparison.transmit(
          text=text, 
          ebn0_db=float(ebn0_db),
          random_seed=random_seed
        )

        # Check if the payload_bits_per_trial and frames_per_trial are consistent across trials
        if payload_bits_per_trial is None:
          payload_bits_per_trial = int(result.original_bits.size)
        if frames_per_trial is None:
          frames_per_trial = int(result.number_of_frames)
        
        # Accumulate the total bit errors, total bits, total frame errors, total frames,
        total_bits += int(result.original_bits.size)
        total_frames += int(result.number_of_frames)
        viterbi_bit_errors += int(result.viterbi.bit_errors)
        neural_bit_errors += int(result.neural.bit_errors)
        viterbi_frame_errors += int(result.viterbi.frame_errors)
        neural_frame_errors += int(result.neural.frame_errors)
        viterbi_text_successes += int(result.viterbi.exact_text_match)
        neural_text_successes += int(result.neural.exact_text_match)

        # Append the latencies for Viterbi and neural decoders
        viterbi_latencies_ms.append(float(result.viterbi.average_latency_ms))
        neural_latencies_ms.append(float(result.neural.average_latency_ms))

        # Print progress
        if (print_progress and (trial_index + 1) % 10 == 0):
          print(f"Eb/N0={ebn0_db:5.1f} dB | Trial {trial_index + 1}/{trials_per_snr}", flush=True)
      
      # Build the aggregate results for Viterbi and neural decoders
      viterbi_result = self._build_decoder_result(
        total_bit_errors=viterbi_bit_errors,
        total_bits=total_bits,
        total_frame_errors=viterbi_frame_errors,
        total_frames=total_frames,
        exact_text_success=viterbi_text_successes,
        total_trials=trials_per_snr,
        latencies_ms=viterbi_latencies_ms
      )
      neural_result = self._build_decoder_result(
        total_bit_errors=neural_bit_errors,
        total_bits=total_bits,
        total_frame_errors=neural_frame_errors,
        total_frames=total_frames,
        exact_text_success=neural_text_successes,
        total_trials=trials_per_snr,
        latencies_ms=neural_latencies_ms
      )

      # Append snr result
      snr_results.append(
        TextSNRResult(
          ebn0_db=float(ebn0_db),
          viterbi=viterbi_result,
          neural=neural_result
        )
      )

      # Print the results for the current SNR value
      elapsed_seconds = perf_counter() - snr_start

      # Print the results for the current SNR value
      print()
      print(f"Completed Eb/N0={ebn0_db:.1f} dB in {elapsed_seconds:.1f} seconds")
      print(f"Viterbi | BER={viterbi_result.ber:.6e} | FER={viterbi_result.frame_error_rate:.4f} | Text success={viterbi_result.exact_text_success_rate:.4f} | Latency={viterbi_result.mean_latency_ms:.3f} ms")
      print(f"Neural  | BER={neural_result.ber:.6e} | FER={neural_result.frame_error_rate:.4f} | Text success={neural_result.exact_text_success_rate:.4f} | Latency={neural_result.mean_latency_ms:.3f} ms")
    
    # Check if payload_bits_per_trial and frames_per_trial are not None
    if payload_bits_per_trial is None or frames_per_trial is None:
      raise ValueError("Payload bits per trial and frames per trial must be determined.")
    
    # Return result
    return TextMonteCarloResult(
      original_text=text,
      payload_bits_per_trial=payload_bits_per_trial,
      frames_per_trial=frames_per_trial,
      trials_per_snr=trials_per_snr,
      snr_results=tuple(snr_results)
    )

# Define function to plot the results of the Monte Carlo trials
def plot_text_monte_carlo_result(result: TextMonteCarloResult, output_directory: str | Path = "results/phase_5") -> None:
  try:
    # Define output directory and create it if it does not exist
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Define arrays for Eb/N0 values, Viterbi and neural decoder metrics
    ebn0_values = np.asarray(
      [point.ebn0_db for point in result.snr_results],  
      dtype=np.float64,
    )
    viterbi_ber = np.asarray( 
      [point.viterbi.ber for point in result.snr_results],
      dtype=np.float64,
    )
    neural_ber = np.asarray(
      [point.neural.ber for point in result.snr_results],
      dtype=np.float64,
    )

    # Define arrays for frame error rates, exact text success rates, 
    # and mean latencies for both decoders
    viterbi_fer = np.asarray(
      [point.viterbi.frame_error_rate for point in result.snr_results],
      dtype=np.float64,
    )
    neural_fer = np.asarray(
      [point.neural.frame_error_rate for point in result.snr_results],
      dtype=np.float64,
    )
    viterbi_success = np.asarray(
      [point.viterbi.exact_text_success_rate for point in result.snr_results],
      dtype=np.float64,
    )
    
    # Define arrays for exact text success rates and mean latencies for both decoders
    neural_success = np.asarray(
      [point.neural.exact_text_success_rate for point in result.snr_results],
      dtype=np.float64,
    )
    viterbi_latency = np.asarray(
      [point.viterbi.mean_latency_ms for point in result.snr_results],
      dtype=np.float64,
    )
    neural_latency = np.asarray(
      [point.neural.mean_latency_ms for point in result.snr_results],
      dtype=np.float64,
    )

    # Calculate the total bits per SNR for determining the minimum measurable BER
    total_bits_per_snr = (result.payload_bits_per_trial * result.trials_per_snr)
    minimum_measurable_ber = (1.0 / total_bits_per_snr)

    # Prepare the BER data for plotting by ensuring that the values are above the minimum measurable BER
    viterbi_ber_plot = np.maximum(viterbi_ber, minimum_measurable_ber)
    neural_ber_plot = np.maximum(neural_ber, minimum_measurable_ber)

    # BER plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(ebn0_values, viterbi_ber_plot, marker="o", label="Soft Viterbi")
    plt.semilogy(ebn0_values, neural_ber_plot, marker="s", label="Neural decoder")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Aggregate bit error rate")
    plt.title("Real-Text BER: Soft Viterbi vs Neural Decoder")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    # Save the BER plot to the output directory
    plt.savefig(output_directory / "text_ber_comparison.png", dpi=300)
    plt.show()

    # Exact text success rate
    plt.figure(figsize=(10, 6))
    plt.plot(ebn0_values, viterbi_success, marker="o", label="Soft Viterbi")
    plt.plot(ebn0_values, neural_success, marker="s", label="Neural decoder")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Exact text success rate")
    plt.title("Probability of Perfect Text Recovery")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the exact text success rate plot to the output directory
    plt.savefig(output_directory / "text_success_rate.png", dpi=300)
    plt.show()

    # Frame error rate
    plt.figure(figsize=(10, 6))
    plt.plot(ebn0_values, viterbi_fer, marker="o", label="Soft Viterbi")
    plt.plot(ebn0_values, neural_fer, marker="s", label="Neural Decoder")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Frame error rate")
    plt.title("Text-Frame Error Rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the frame error rate plot to the output directory
    plt.savefig(output_directory / "text_frame_error_rate.png", dpi=300)
    plt.show()

    # Latency plot
    plt.figure(figsize=(10, 6))
    plt.plot(ebn0_values, viterbi_latency, marker="o", label="Soft Viterbi")
    plt.plot(ebn0_values, neural_latency, marker="s", label="Neural Decoder")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Mean decoding latency (ms/frame)")
    plt.title("Text Decoder Latency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the latency plot to the output directory
    plt.savefig(output_directory / "text_decoder_latency.png", dpi=300)
    plt.show()
  except CustomException as e:
    raise CustomException(e, sys)