#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import time
from typing import Generator

import sentencepiece
from mlx import core as mx

from mlx_examples.open_elm import open_elm

# From "The Treasure Island" by R.L.Stevenson, public domain.
PROMPT = (
    "Squire Trelawney, Dr. Livesey, and the rest of these gentlemen having "
    "asked me to write down the whole particulars about Treasure Island, "
    "from the"
)


def generate_token(
    prompt_tokens: mx.array, model: open_elm.OpenELM, sampling_temperature: float = 0.0
) -> Generator[int, None, None]:
    """Generates a single output token at a time for a given set of prompt tokens.

    Args:
        prompt_tokens: Tokenized prompt from SentencePiece tokenizer.
        model: An OpenELM model instance used for generating text completions.
        sampling_temperature: A float specifying the sampling temperature for generation,
            which affects the randomness of the generated text. A value of 0 (default) means
            deterministic output, while higher values introduce more randomness.

    Returns:
        A generator object which produces one output token at a time.
    """

    def sample(logits):
        if sampling_temperature == 0:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits * (1.0 / sampling_temperature))

    # Process the prompt.
    output = model(
        {
            "input_ids": prompt_tokens,
            "past_key_values": None,
            "use_kv_cache": True,
            "is_causal": True,
        }
    )
    cache = output["past_key_values"]
    logits = output["logits"]
    y = sample(logits[:, -1])
    yield y

    while True:
        output = model(
            {
                "input_ids": y[:, None],
                "past_key_values": cache,
                "use_kv_cache": True,
                "is_causal": True,
            }
        )
        cache = output["past_key_values"]
        logits = output["logits"]
        y = sample(logits.squeeze(1))
        yield y


def generate(
    model: open_elm.OpenELM,
    tokenizer: sentencepiece.SentencePieceProcessor,
    prompt: str,
    max_tokens: int,
    sampling_temperature: float = 0.0,
    print_output: bool = False,
):
    """Generates and prints a response for a given prompt.

    Args:
        model: An OpenELM model instance used for generating text completions.
        tokenizer: A SentencePieceProcessor instance for tokenizing text for the model.
        prompt: The initial text prompt to generate completions for.
        max_tokens: The maximum number of tokens to generate for the completion.
        sampling_temperature: A float specifying the sampling temperature for generation,
            which affects the randomness of the generated text. A value of 0 (default) means
            deterministic output, while higher values introduce more randomness.

    Returns:
        None. The function directly prints the generated text completion to the standard output.
    """
    tokenized_prompt = mx.array([[tokenizer.bos_id()] + tokenizer.Encode(prompt)])
    num_prompt_tokens = len(tokenized_prompt.flatten())
    print(f"{num_prompt_tokens} token(s) in the prompt.")

    # Evaluation is done lazily, graph is built for several tokens each time,
    # except for the first token after the prompt. The first token is evaluated
    # in order to measure the prompt processing throughput.
    tokens_per_eval = 8
    start = time.perf_counter()
    tokens = []

    # Warm up the model.
    _ = generate_token(tokenized_prompt, model, sampling_temperature)

    generated_tokens = 0
    elapsed_prompt = 0.0
    for token, ntoks in zip(
        generate_token(tokenized_prompt, model, sampling_temperature),
        range(max_tokens),
    ):
        tokens.append(token)
        if ntoks == 0:
            mx.eval(tokens)
            elapsed_prompt = time.perf_counter() - start

        if (len(tokens) % tokens_per_eval) == 0:
            mx.eval(tokens)
            if print_output:
                s = tokenizer.Decode([t.item() for t in tokens])
                print(s, end="", flush=True)
            tokens = []
        generated_tokens += 1

    mx.eval(tokens)
    if print_output:
        s = tokenizer.Decode([t.item() for t in tokens])
        print(s, flush=True)

    elapsed_total = time.perf_counter() - start
    elapsed_generation = elapsed_total - elapsed_prompt
    prompt_tps = tokenized_prompt.size / elapsed_prompt
    generation_tps = generated_tokens / elapsed_generation
    total_tps = (generated_tokens + num_prompt_tokens) / elapsed_total

    print(
        f"Throughput: prompt {prompt_tps:.2f} t/s, generation {generation_tps:.2f} t/s. "
        f"{generated_tokens} tokens generated in {elapsed_total - elapsed_prompt}s. "
        f"Total throughput: {total_tps} t/s."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test loading checkpoints and inference for MLX port of Apple OpenELM LLM."
    )
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        required=True,
        help="Path to MLX model directory containing model weights, JSON config and the "
        "tokenizer file.",
    )
    parser.add_argument("--prompt", default=PROMPT, help="Prompt for inference.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--sampling-temperature", type=float, default=0.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="When specified, output will be printed to console.",
    )

    args = parser.parse_args()

    assert args.sampling_temperature >= 0.0, args.sampling_temperature
    assert args.max_tokens > 0, args.max_tokens

    model, tokenizer = open_elm.load_model(args.model_dir)

    print("Prompt:", args.prompt)

    generate(
        model,
        tokenizer,
        args.prompt,
        args.max_tokens,
        sampling_temperature=args.sampling_temperature,
        print_output=args.print_output,
    )
