"""
Run a text payload comparison between Soft Viterbi and the neural decoder.
"""
import sys
import numpy as np

from src.ml.evaluation.mc_trial import TextMonteCarloEvaluator,plot_text_monte_carlo_result
from src.ml.evaluation.text_decoder_comparison import TextDecoderComparison
from src.exception.exception import CustomException

# Define function to first snr reaching target
def first_snr_reaching_target(snr_results, decoder_name: str, target: float) -> float | None:
  for point in snr_results:
    decoder = getattr(point, decoder_name)
    if (decoder.exact_text_success_rate >= target):
      return point.ebn0_db
  
  # Return None if no SNR point reaches the target success rate
  return None


# Define function to run the text decoder comparison
def main() -> None:
  try:
    # Define message to transmit
    message = input("Enter a message to transmit: ")

    # Define comparison parameters
    evaluator = TextMonteCarloEvaluator(
      neural_model_path="results/models/best_decoder.keras"
    )

    # Iterate using different SNR levels
    result = evaluator.evaluate(
      text=message,
      ebn0_values_db=np.array([
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
      ], dtype=np.float64),
      trials_per_snr=200,
      base_random_seed=42,
      print_progress=True
    )

    # Plot the results of the text decoder comparison
    plot_text_monte_carlo_result(result, output_directory="results/phase_5")

    # Print the first SNR point where each decoder reaches a 95% exact text success rate
    viterbi_snr_90 = first_snr_reaching_target(result.snr_results, "viterbi", 0.90)
    neural_snr_90 = first_snr_reaching_target(result.snr_results, "neural", 0.90)
    print(f"Viterbi decoder reaches 90% success rate at SNR: {viterbi_snr_90} dB")
    print(f"Neural decoder reaches 90% success rate at SNR: {neural_snr_90} dB")
  except Exception as e:
    raise CustomException(e, sys)

# Run the main function if this script is executed directly
if __name__ == "__main__":
  main()