from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from data_utils import BaseDataCollator, BaseDataset, postprocess_decoder_output


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    generation_kwargs: dict[str | Any],
    output_with_prompt: bool = False,
    batch_size: int = 2,
):
    dataset = BaseDataset(prompts)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=BaseDataCollator(tokenizer),
        shuffle=False,
    )

    result = []
    for prompt, inputs in data_loader:
        with torch.no_grad():
            outputs: GenerateDecoderOnlyOutput = model.generate(
                **inputs, **generation_kwargs, return_dict_in_generate=True
            )
        if output_with_prompt:
            result += postprocess_decoder_output(outputs, prompt)
        else:
            result += postprocess_decoder_output(outputs)

    return result


def call(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    output_with_prompt: bool = False,
    generator: bool = False,
    postprocess: bool = False,
    batch_size: int = 2,
):
    dataset = BaseDataset(prompts)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=BaseDataCollator(tokenizer),
        shuffle=False,
    )

    result = []
    for prompt, inputs in data_loader:
        position_ids = inputs["attention_mask"].cumsum(axis=-1) - 1
        position_ids.masked_fill_(inputs["attention_mask"] == 0, 1)
        with torch.no_grad():
            outputs: CausalLMOutputWithCrossAttentions = model(
                **inputs, position_ids=position_ids
            )

        if generator:
            if postprocess:
                if output_with_prompt:
                    yield postprocess_decoder_output(outputs, prompt)
                else:
                    yield postprocess_decoder_output(outputs)
            elif output_with_prompt:
                yield outputs, prompt
            else:
                yield outputs

        if postprocess:
            if output_with_prompt:
                result += postprocess_decoder_output(outputs, prompt)
            else:
                result += postprocess_decoder_output(outputs)

    return result
