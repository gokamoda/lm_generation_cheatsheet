import json
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from utils.mylogger import init_logging

logger = init_logging(__name__, clear=True)


class BaseDataset(Dataset):
    prompts: list[str]

    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class BaseDataCollator:
    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(self, prompts: list[str]) -> tuple[list[str], dict]:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest")
        return prompts, inputs


def load_data(file_path: str | Path):
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if file_path.suffix == ".csv":
        df = pl.read_csv(file_path)
    elif file_path.suffix == ".jsonl":
        df = pl.read_ndjson(file_path)

    return df


def postprocess_decoder_output(
    output: GenerateDecoderOnlyOutput | CausalLMOutputWithCrossAttentions,
    past_key_values=False,
    prompts: list[str] = None,
):
    processed_output = []
    process_keys = list(output.keys())

    if not past_key_values:
        process_keys.remove("past_key_values")

    for i in range(len(output[process_keys[0]])):
        record = {k: output[k][i] for k in process_keys}
        if prompts:
            record["prompt"] = prompts[i]
        processed_output.append(record)

    return processed_output
