import sys
from argparse import ArgumentParser

import torch
from torchtyping import TensorType
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation import call
from utils.mylogger import init_logging

logger = init_logging(__name__, clear=True)


def compute_sequence_prob(
    model_name_or_path: str,
    prompt: list[str],
    sequence: list[str],
    batch_size: int = 2,
) -> torch.Tensor:
    """computes p(sequence|prompt) for each pair of prompt and sequence

    Parameters
    ----------
    model_name_or_path : str
    prompt : list[str]
    sequence : list[str]
    batch_size : int, optional
        by default 2

    Returns
    -------
    torch.Tensor
        tensor of p(sequence|prompt) for each pair of prompt and sequence
    """

    if sys.platform == "darwin":
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    end_token_id = tokenizer.eos_token_id

    full_prompts = [f"{p}{s}" for p, s in zip(prompt, sequence)]
    sequence_probs = []
    for result, prompts in call(
        model=model,
        tokenizer=tokenizer,
        prompts=full_prompts,
        generator=True,
        postprocess=False,
        output_with_prompt=True,
        batch_size=batch_size,
    ):
        sequences: list[
            list[int]
        ]  # list of token ids for each sequence with end token at the end
        sequences = [
            tokenizer.encode(sequence[full_prompts.index(p)], add_special_tokens=False)
            + [end_token_id]
            for p in prompts
        ]

        # get max sequence length to reduce computation
        max_sequence_length: int = max([len(s) for s in sequences])

        # probabilities of next token prediction
        probs: TensorType[len(sequences), max_sequence_length, model.vocab_size]
        probs = torch.nn.functional.softmax(
            result["logits"][:, -max_sequence_length:, :], dim=-1
        )

        # positions to retrieve from probs
        padded_sequences: TensorType[len(sequences), max_sequence_length]
        padded_sequences = torch.stack(
            [
                torch.nn.functional.pad(
                    torch.tensor(s), (max_sequence_length - len(s), 0), value=-1
                )
                for s in sequences
            ]
        )

        # mask for padding
        mask: TensorType[len(sequences), max_sequence_length]
        mask = padded_sequences != -1

        # replace padding with 0
        padded_sequences[padded_sequences == -1] = 0

        # get probabilities of next token
        probs: TensorType[len(sequences), max_sequence_length] = torch.gather(
            probs, -1, padded_sequences.unsqueeze(-1)
        ).squeeze()

        # apply mask and compute product of probabilities
        sequence_probs.append(torch.prod(probs * mask + ~mask, dim=-1))

    return torch.cat(sequence_probs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--prompt", type=str, required=True)
    # parser.add_argument("--sequence", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    prompts = [
        "<|endoftext|> Tokyo is the capital of",
        "<|endoftext|> Washington D.C. is the capital of",
    ]
    sequences = [" Japan.", " the United States."]
    sequence_probs = compute_sequence_prob(
        model_name_or_path=args.model,
        prompt=prompts,
        sequence=sequences,
        batch_size=args.batch_size,
    )

    for p, s, prob in zip(prompts, sequences, sequence_probs):
        logger.info(f'p("{s}"|"{p}") = {prob}')
