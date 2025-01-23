from functools import partial

from torchtyping import TensorType
from transformers import AutoTokenizer

from hfdecoderutils.model_with_cache import (
    BaseDataCollator,
    BaseInputs,
    BaseInstance,
    BatchGenerationResult,
    DeterministicModelWithCache,
)
from hfdecoderutils.utils.mytorchtyping import BATCH_SIZE, VOCAB


def postprocess_simple_generation(
    tokenizer: AutoTokenizer,
    batch_result: BatchGenerationResult,
    inputs: list[BaseInputs],
) -> list[BaseInstance]:
    decoded = tokenizer.batch_decode(
        batch_result.outputs.sequences, skip_special_tokens=False
    )
    instances = [
        BaseInstance(
            input_kwargs=batch_result.inputs[i].__dict__,
            generation_kwargs=batch_result.generation_kwargs,
            output_kwargs={
                "generated_sequence": decoded[i].split(inputs[i].prompt)[-1]
            },
        )
        for i in range(len(batch_result.inputs))
    ]
    for i in range(len(batch_result.inputs)):
        del instances[i].instance_id
    return instances


def test_model_with_cache():
    model_bame = "gpt2"

    model = DeterministicModelWithCache(
        model_name_or_path=model_bame,
    )
    prompts = [
        " Tokyo is the capital of",
        " Beijing is the capital of",
        " Paris is the capital of",
        " Berlin is the capital of",
        " Berlin is the capital of",
    ]

    result = model.run_inference(
        cache_file_name="test-model-with-cache",
        generation_kwargs={
            "return_dict_in_generate": True,
            "do_sample": False,
            "pad_token_id": model.tokenizer.eos_token_id,
            "max_new_tokens": 6,
            "output_logits": True,
        },
        inputs=[BaseInputs(prompt=prompt) for prompt in prompts],
        collator=BaseDataCollator(model.tokenizer),
        batch_size=1,
        post_procecss_fn=partial(postprocess_simple_generation, model.tokenizer),
    )
    print(result)


def postprocess_get_label_probs(
    tokenizer: AutoTokenizer,
    labels: list[str],
    batch_result: BatchGenerationResult,
    inputs: list[BaseInputs],
) -> list[BaseInstance]:
    logits: TensorType[BATCH_SIZE, VOCAB] = batch_result.outputs.logits[0]
    labels_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]
    logits = logits[:, labels_ids]
    probs = logits.softmax(dim=-1)

    instances = [
        BaseInstance(
            input_kwargs=batch_result.inputs[i].__dict__,
            generation_kwargs=batch_result.generation_kwargs,
            output_kwargs={"probabilities": probs[i].tolist()},
        )
        for i in range(len(batch_result.inputs))
    ]

    for i in range(len(batch_result.inputs)):
        del instances[i].instance_id

    return instances


def test_model_with_cache_prob():
    model_bame = "gpt2"

    model = DeterministicModelWithCache(
        model_name_or_path=model_bame,
    )
    prompts = [
        " Tokyo is the capital of",
        " Beijing is the capital of",
        " Paris is the capital of",
        " Berlin is the capital of",
        " Berlin is the capital of",
    ]
    labels = [" Japan", " China", " France", " Germany"]

    result = model.run_inference(
        cache_file_name="test-model-with-cache-prob_" + "-".join(labels),
        generation_kwargs={
            "return_dict_in_generate": True,
            "do_sample": False,
            "pad_token_id": model.tokenizer.eos_token_id,
            "max_new_tokens": 1,
            "output_logits": True,
        },
        inputs=[BaseInputs(prompt=prompt) for prompt in prompts],
        collator=BaseDataCollator(model.tokenizer),
        batch_size=1,
        post_procecss_fn=partial(postprocess_get_label_probs, model.tokenizer, labels),
    )
    print(result)


if __name__ == "__main__":
    test_model_with_cache()
    # test_model_with_cache_prob()
