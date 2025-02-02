import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Generator

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType
from tqdm import tqdm as std_tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.tokenization_utils import BatchEncoding

from .utils import init_logging
from .utils.int_utils import check_power_of_2
from .utils.mytorchtyping import BATCH_SIZE

tqdm = partial(std_tqdm, dynamic_ncols=True)

logger = init_logging(__name__, clear=True)

LOG_PATH = "latest.log"
logger = init_logging(__name__, log_path=LOG_PATH)


@dataclass
class BaseInputs:
    prompt: str

    def __init__(self, prompt: str, **kwargs):
        self.prompt = prompt

    def get_prompt(self) -> str:
        self.prompt = self.prompt
        return self.prompt

    def __len__(self):
        return len(self.prompt)


class BaseDataset(Dataset):
    prompts: list[str]

    def __init__(self, inputs: list[BaseInputs]):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class BaseDataCollator:
    def __init__(
        self,
        tokenizer,
    ):
        self.tokenizer = tokenizer

    def __call__(self, inputs: list[BaseInputs]) -> tuple[list[str], dict]:
        prompts = [_input.get_prompt() for _input in inputs]
        tokenizer_output = self.tokenizer(
            prompts, return_tensors="pt", padding="longest"
        )
        return inputs, tokenizer_output


@dataclass
class BaseInstance:
    def __init__(
        self, input_kwargs: dict, generation_kwargs, output_kwargs: dict[str] = None
    ):
        for fields in [input_kwargs, output_kwargs, generation_kwargs]:
            if isinstance(fields, dict):
                for key, value in fields.items():
                    setattr(self, key, value)
            elif fields is None:
                continue
            else:
                raise ValueError

    def __repr__(self):
        lines = [f"\t{key}: {value}" for key, value in self.__dict__.items()]
        lines = [f"{self.__class__.__name__}("] + lines + [")"]
        return "\n".join(lines)

    def to_json(self):
        return self.__dict__


@dataclass
class BatchGenerationResult:
    inputs: list[BaseInputs]
    model_inputs: BatchEncoding
    generation_kwargs: dict[str, Any]
    outputs: GenerateDecoderOnlyOutput


def get_generation_cache_dir(model_name_or_path: str) -> Path:
    directory = Path("./work").joinpath(
        f"generation_cache/{model_name_or_path.split('/')[-1]}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_of_dict_diff(set1: list[dict[str, Any]], set2: list[dict[str, Any]]):
    keys = list(set1[0].keys())
    set1_as_tuple = [tuple(d[key] for key in keys) for d in set1]
    set2_as_tuple = [tuple(d[key] for key in keys) for d in set2]

    diff = set(set1_as_tuple) - set(set2_as_tuple)

    if diff:
        return [dict(zip(keys, d)) for d in diff]
    return []


def is_chat_model(model_name_or_path: str) -> bool:
    return (
        "chat" in model_name_or_path
        or "Instruct" in model_name_or_path
        or "it" in model_name_or_path
    )


class DeterministicModelWithCache:
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": True,
        "output_logits": True,
    }

    def __init__(
        self,
        model_name_or_path: str,
        model: AutoModelForCausalLM = None,
        tokenizer: AutoTokenizer = None,
        cache_dir: Path = None,
    ):
        if cache_dir is None:
            self.cache_dir = get_generation_cache_dir(model_name_or_path)
        else:
            self.cache_dir = cache_dir

        self.model_name_or_path = model_name_or_path
        self.model = model

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, padding_side="left"
            )
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def load_model(self, tensor_type=None):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path, device_map="auto"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path
                ).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)

    def get_precomputed_results(
        self,
        inputs: list[BaseInputs],
        cache_path: Path,
        generation_kwargs: dict = None,
        output_fields: list[str] = None,
    ) -> tuple[list[str], pl.DataFrame]:
        instances = [
            BaseInstance(_input.__dict__, generation_kwargs) for _input in inputs
        ]
        df_queries = pl.DataFrame([instance.to_json() for instance in instances])
        df_queries = df_queries.unique(subset=df_queries.columns)
        precomputed_results = pl.DataFrame()

        if cache_path.exists():
            try:
                precomputed_results = pl.read_ndjson(cache_path)
                precomputed_results = df_queries.join(
                    precomputed_results, on=df_queries.columns, how="inner"
                )

                precomputed_results = precomputed_results.with_columns(
                    pl.Series(["True" for _ in range(len(precomputed_results))]).alias(
                        "is_precomputed"
                    )
                )

                input_colums = df_queries.columns.copy()

                df_queries = df_queries.join(
                    precomputed_results, on=df_queries.columns, how="left"
                )

                precomputed_results = df_queries.filter(
                    pl.col("is_precomputed").is_not_null()
                )

                df_queries = (
                    df_queries.filter(pl.col("is_precomputed").is_null())
                    .select(input_colums)
                    .unique(subset=input_colums)
                )

                inputs_to_compute = [
                    inputs[0].__class__(**row) for row in df_queries.to_dicts()
                ]
            except Exception as e:
                logger.error(e)
                precomputed_results = pl.DataFrame()

        inputs_to_compute = [
            inputs[0].__class__(**row) for row in df_queries.to_dicts()
        ]

        return inputs_to_compute, precomputed_results

    def save_new_results(self, new_results: list[BaseInstance], cache_path: Path):
        new_results = [instance.to_json() for instance in new_results]
        if not cache_path.exists():
            new_results_df = pl.DataFrame(new_results)
            new_results_df.write_ndjson(cache_path)
            return new_results_df
        try:
            pre_computed_results = pl.read_ndjson(cache_path)
        except Exception as e:
            logger.error(e)
            new_results_df = pl.DataFrame(new_results)
            new_results_df.write_ndjson(cache_path)
            return new_results_df

        new_results_df = pl.DataFrame(new_results)
        pl.concat([pre_computed_results, new_results_df], how="vertical").write_ndjson(
            cache_path
        )
        return new_results_df

    def model_generate(self, model_input_kwargs, generation_kwargs):
        model_input_kwargs = model_input_kwargs.to(self.model.device)
        return self.model.generate(**model_input_kwargs, **generation_kwargs)

    def batch_generate(
        self,
        inputs: list[BaseInputs],
        collator,
        generation_kwargs: dict,
        batch_size: int,
    ) -> Generator[BatchGenerationResult, None, None]:
        logger.info(f"Batch size: {batch_size}\nNum prompts to process: {len(inputs)}")

        data_loader = DataLoader(
            dataset=BaseDataset(
                sorted(
                    inputs,
                    key=len,
                    reverse=True,
                )
            ),
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
        )

        with torch.no_grad():
            for _inputs, _model_inputs in tqdm(data_loader):
                for _ in range(10):
                    try:
                        outputs = self.model_generate(_model_inputs, generation_kwargs)

                        yield BatchGenerationResult(
                            inputs=_inputs,
                            model_inputs=_model_inputs,
                            outputs=outputs,
                            generation_kwargs=generation_kwargs,
                        )
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        raise e
                    except RuntimeError as e:
                        logger.error(
                            f"{e}\nsampling error. Trying again with smaller batch size"
                        )

    def set_instance_ids(self, inputs: list[BaseInputs]):
        for i, _input in enumerate(inputs):
            _input.instance_id = i

        return inputs

    def remove_instance_ids(self, inputs: list[BaseInputs]):
        for _input in inputs:
            del _input.instance_id

        return inputs

    def dynamic_batch_generate(
        self,
        inputs: list[BaseInputs],
        generation_kwargs: dict,
        collator: BaseDataCollator,
        starting_batch_size: int = 32,
    ) -> Generator[BatchGenerationResult, None, None]:
        assert isinstance(
            starting_batch_size, int
        ), "Starting batch size must be an integer"
        assert starting_batch_size > 0, "Starting batch size must be greater than 0"
        assert check_power_of_2(
            starting_batch_size
        ), "Starting batch size must be a power of 2"

        batch_size = starting_batch_size
        if "do_sample" in generation_kwargs:
            if not generation_kwargs["do_sample"]:
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None

        inputs = self.set_instance_ids(inputs)
        pop_offsets = torch.zeros(len(inputs), dtype=torch.int32)
        while True:
            try:
                batch_result_generator = self.batch_generate(
                    inputs=inputs,
                    collator=collator,
                    generation_kwargs=generation_kwargs,
                    batch_size=batch_size,
                )
                for batch_id, batch_result in enumerate(batch_result_generator):
                    batch_result: BatchGenerationResult
                    yield batch_result

                    for _input in batch_result.inputs:
                        inputs.pop(_input.instance_id - pop_offsets[_input.instance_id])
                        pop_offsets[_input.instance_id + 1 :] += 1

                    if len(inputs) == 0:
                        batch_result_generator.close()
                        torch.cuda.empty_cache()
                        return

                    num_input_tokens: TensorType[BATCH_SIZE] = (
                        batch_result.model_inputs["attention_mask"].sum(dim=-1)
                    )
                    if batch_id == 0:
                        max_tokens = int(num_input_tokens.max())

                    if check_power_of_2(batch_size):
                        if int(num_input_tokens.min()) * 1.5 < max_tokens:
                            batch_size = int(batch_size * 1.5)
                            batch_result_generator.close()
                            break
                    elif int(num_input_tokens.min()) / 1.5 * 2 < max_tokens:
                        batch_size = int(batch_size / 1.5 * 2)
                        batch_result_generator.close()
                        break

            except torch.cuda.OutOfMemoryError:
                batch_result_generator.close()
                torch.cuda.empty_cache()

                if batch_size == 1:
                    raise Exception("Batch size is 1 and still out of memory") from None
                if check_power_of_2(batch_size):
                    batch_size = int(batch_size // 2 * 1.5)
                else:
                    batch_size = int(batch_size // 1.5)

        batch_result_generator.close()
        torch.cuda.empty_cache()

    def run_inference(
        self,
        inputs: list[BaseInputs],
        generation_kwargs: dict,
        collator: BaseDataCollator,
        post_procecss_fn: callable,
        cache_file_name: str,
        batch_size: int = 32,
    ):
        cache_file_path = self.cache_dir.joinpath(f"{cache_file_name}.jsonl")

        inputs_to_compute, processed_result = self.get_precomputed_results(
            inputs=inputs,
            cache_path=cache_file_path,
            generation_kwargs=generation_kwargs,
            output_fields=["generated_sequences"],
        )

        if not inputs_to_compute:
            return processed_result
        elif self.model is None:
            self.load_model()

        for batch_result in self.dynamic_batch_generate(
            generation_kwargs=generation_kwargs,
            collator=collator,
            starting_batch_size=batch_size,
            inputs=inputs_to_compute,
        ):
            self.save_new_results(
                post_procecss_fn(batch_result, inputs_to_compute), cache_file_path
            )

        _, processed_result = self.get_precomputed_results(
            inputs=inputs,
            cache_path=cache_file_path,
            generation_kwargs=generation_kwargs,
            output_fields=["generated_sequences"],
        )
        return processed_result

    def p_sequence_given_prompt(
        self, prompts: list[str], sequences: list[str], suffix: str = None
    ):
        cache_file_name_suffix = ""
        if suffix:
            # copy current cache file to new cache file with prefix
            cache_file_name_suffix = f"_{suffix}"
            # copyfile(
            #     self.cache_dir.joinpath("p_sequence_given_prompt.jsonl"),
            #     self.cache_dir.joinpath(f"p_sequence_given_prompt{cache_file_name_suffix}.jsonl"),
            # )

        cache_path = self.cache_dir.joinpath(
            f"p_sequence_given_prompt{cache_file_name_suffix}.jsonl"
        )

        inputs_to_compute, precomputed_result = self.get_precomputed_results(
            inputs=[
                {"prompt": prompt, "sequence": sequence}
                for prompt, sequence in zip(prompts, sequences)
            ],
            cache_path=cache_path,
        )

        if not inputs_to_compute:
            return precomputed_result

        if self.model is None:
            self.load_model()
        prompts = [instance["prompt"] for instance in inputs_to_compute]
        sequences = [instance["sequence"] for instance in inputs_to_compute]
        full_prompts = [
            f"{instance['prompt']}{instance['sequence']}"
            for instance in inputs_to_compute
        ]

        batch_size = self.batch_size

        while True:
            try:
                data_loader = DataLoader(
                    dataset=BaseDataset(full_prompts),
                    batch_size=int(batch_size / 2),
                    collate_fn=BaseDataCollator(self.tokenizer),
                    shuffle=False,
                )

                sequence_probs = []
                sequence_probs_no_eos = []
                with torch.no_grad():
                    for _prompts, inputs in tqdm(data_loader):
                        position_ids = inputs["attention_mask"].cumsum(axis=-1) - 1
                        position_ids.masked_fill_(inputs["attention_mask"] == 0, 1)

                        outputs = self.model(**inputs)

                        _sequences: list[
                            list[int]
                        ]  # list of token ids for each sequence with end token at the end
                        _sequences = [
                            self.tokenizer.encode(
                                sequences[full_prompts.index(p)],
                                add_special_tokens=False,
                            )
                            + [self.tokenizer.eos_token_id]
                            for p in _prompts
                        ]

                        # get max sequence length to reduce computation
                        max_sequence_length: int = max([len(s) for s in _sequences])

                        # probabilities of next token prediction
                        probs: TensorType[
                            len(_sequences), max_sequence_length, self.model.vocab_size
                        ]
                        probs = torch.nn.functional.softmax(
                            outputs["logits"][:, -max_sequence_length:, :], dim=-1
                        )

                        # positions to retrieve from probs
                        padded_sequences: TensorType[
                            len(_sequences), max_sequence_length
                        ]
                        padded_sequences = torch.stack(
                            [
                                torch.nn.functional.pad(
                                    torch.tensor(s),
                                    (max_sequence_length - len(s), 0),
                                    value=-1,
                                )
                                for s in _sequences
                            ]
                        )

                        # mask for padding
                        mask: TensorType[len(_sequences), max_sequence_length]
                        mask = padded_sequences != -1

                        mask_noeos = mask.clone()
                        mask_noeos[:, -1] = 0

                        # replace padding with 0
                        padded_sequences[padded_sequences == -1] = 0

                        # get probabilities of next token
                        probs: TensorType[len(_sequences), max_sequence_length] = (
                            torch.gather(
                                probs, -1, padded_sequences.unsqueeze(-1)
                            ).squeeze()
                        )
                        sequence_probs += torch.prod(
                            probs * mask + ~mask, dim=-1
                        ).tolist()
                        sequence_probs_no_eos += torch.prod(
                            probs * mask_noeos + ~mask_noeos, dim=-1
                        ).tolist()
                break
            except (torch.OutOfMemoryError, RuntimeError):
                if batch_size == 2:
                    logger.error(
                        "Out of memory error even with batch size 1. Please reduce the sequence length."
                    )
                    break
                batch_size = int(batch_size / 2)
                logger.error(
                    f"Out of memory error. Reducing batch size to {batch_size} and trying again."
                )

        new_results = [
            {
                "prompt": prompt,
                "sequence": sequence,
                "prob": prob,
                "prob_no_eos": prob_no_eos,
            }
            for prompt, sequence, prob, prob_no_eos in zip(
                prompts, sequences, sequence_probs, sequence_probs_no_eos
            )
        ]
        new_result_df = self.save_new_results(new_results, cache_path=cache_path)
        if precomputed_result.height == 0:
            result = new_result_df
        else:
            result = pl.concat([precomputed_result, new_result_df], how="vertical")

        return result

    def create_random_sequences(self, length: int, num_sequences: int) -> list[str]:
        ids = list(self.tokenizer.get_vocab().values())
        sequences = []

        for i in range(num_sequences):
            random.seed(42 + i)
            sequences.append(
                self.tokenizer.decode(
                    random.choices(
                        ids,
                        k=length,
                    )
                )
            )

        return sequences
