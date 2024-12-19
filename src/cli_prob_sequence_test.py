from argparse import ArgumentParser

import torch

from cli_prob_sequence import compute_sequence_prob
from utils.mylogger import init_logging

logger = init_logging(__name__, clear=True)


def test_compute_sequence_prob(model: str = None):
    if model is None:
        model = "gpt2"
    prompts = [
        "<|endoftext|> Tokyo is the capital of",
        "<|endoftext|> Washington D.C. is the capital of",
    ]
    sequences = [" Japan.", " the ch a asfa."]

    sequence_probs_bs1 = compute_sequence_prob(
        model_name_or_path=model,
        prompt=prompts,
        sequence=sequences,
        batch_size=1,
    )

    logger.info(sequence_probs_bs1)

    sequence_probs_bs2 = compute_sequence_prob(
        model_name_or_path=model,
        prompt=prompts,
        sequence=sequences,
        batch_size=2,
    )

    logger.info(sequence_probs_bs2)

    assert torch.allclose(
        sequence_probs_bs1, sequence_probs_bs2
    ), f"Not equivalent results for batch size 1 and 2 on model {model}"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    test_compute_sequence_prob(model=args.model)
