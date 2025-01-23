from dataclasses import dataclass
from functools import partial

import torch
from model_with_cache import (
    BaseDataCollator,
    BaseInputs,
    BaseInstance,
    BatchGenerationResult,
    DeterministicModelWithCache,
)
from torchtyping import TensorType


class ModelCallWithCache(DeterministicModelWithCache):
    def model_generate(self, model_input_kwargs, generation_kwargs):
        return self.model(**model_input_kwargs)


class CollatorWithPositionIds(BaseDataCollator):
    def __call__(self, inputs: list[BaseInputs]) -> tuple[list[str], dict]:
        inputs, tokenizer_output = super().__call__(inputs)
        position_ids = tokenizer_output["attention_mask"].cumsum(axis=-1) - 1
        position_ids.masked_fill_(tokenizer_output["attention_mask"] == 0, 1)
        tokenizer_output["position_ids"] = position_ids

        return inputs, tokenizer_output


@dataclass
class SeqProbInputs(BaseInputs):
    prompt: str
    prefix: str
    sequence: str

    def __init__(self, prefix: str, sequence: str):
        self.prefix = prefix
        self.sequence = sequence

    def get_prompt(self):
        self.prompt = self.prefix + self.sequence
        return self.prompt

    def __len__(self):
        return len(self.prefix + self.sequence)


def postprocess_prob_generation(
    tokenizer, batch_result: BatchGenerationResult, inputs: list[BaseInputs]
) -> list[BaseInstance]:
    sequence_probs = []
    sequence_probs_no_eos = []
    sequence_probs = []
    sequences_ids: list[list[int]] = [
        tokenizer.encode(
            _input.sequence,
            add_special_tokens=False,
        )
        + [tokenizer.eos_token_id]
        for _input in batch_result.inputs
    ]

    max_sequence_length: int = max([len(s) for s in sequences_ids])

    # probabilities of next token prediction
    probs: TensorType[len(sequences_ids), max_sequence_length, tokenizer.vocab_size] = (
        torch.nn.functional.softmax(
            batch_result.outputs["logits"][:, -max_sequence_length:, :], dim=-1
        )
    )

    # positions to retrieve from probs
    padded_sequences: TensorType[len(sequences_ids), max_sequence_length] = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor(s),
                (max_sequence_length - len(s), 0),
                value=-1,
            )
            for s in sequences_ids
        ]
    )

    # mask for padding
    mask: TensorType[len(sequences_ids), max_sequence_length] = padded_sequences != -1
    mask_noeos = mask.clone()
    mask_noeos[:, -1] = 0

    # replace padding with 0
    padded_sequences[padded_sequences == -1] = 0

    # get probabilities of next token
    probs: TensorType[len(sequences_ids), max_sequence_length] = torch.gather(
        probs, -1, padded_sequences.unsqueeze(-1)
    ).squeeze()

    sequence_probs += torch.prod(probs * mask + ~mask, dim=-1).tolist()
    sequence_probs_no_eos += torch.prod(
        probs * mask_noeos + ~mask_noeos, dim=-1
    ).tolist()

    instances = [
        BaseInstance(
            input_kwargs=batch_result.inputs[i].__dict__,
            generation_kwargs=batch_result.generation_kwargs,
            output_kwargs={
                "probs": sequence_probs[i],
                "probs_no_eos": sequence_probs_no_eos[i],
            },
        )
        for i in range(len(batch_result.inputs))
    ]
    for i in range(len(batch_result.inputs)):
        del instances[i].instance_id
    return instances


def test_p_sequence_given_prompt():
    model_bame = "gpt2"

    model = ModelCallWithCache(
        model_name_or_path=model_bame,
    )
    prompts = [
        " Tokyo is the capital of",
        " Washington D.C. is the capital of",
        " Paris is the capital of",
        " Berlin is the capital of",
    ]
    sequences = [
        " Japan",
        " the United States",
        " France",
        " Germany",
    ]

    result = model.run_inference(
        cache_file_name="test-model-call",
        generation_kwargs={},
        inputs=[
            SeqProbInputs(prefix=prompt, sequence=sequence)
            for prompt, sequence in zip(prompts, sequences)
        ],
        collator=CollatorWithPositionIds(model.tokenizer),
        batch_size=4,
        post_procecss_fn=partial(postprocess_prob_generation, model.tokenizer),
    )
    print(result)


if __name__ == "__main__":
    test_p_sequence_given_prompt()
