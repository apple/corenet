#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""
Evaluate commonsense reasoning performance on 8 commonsense reasoning tasks,
collectively called "CommonSense 170k" in LLM-Adapters <https://arxiv.org/abs/2304.01933>.

Code is adapted from https://github.com/AGI-Edgerunners/LLM-Adapters

In addition to the generation-style evaluation used by LLM-Adapters, we add a
multiple-choice-style evaluation. This massively improves results for small
models.

Currently, this only supports the LLama tokenizer. We may add support for
other tokenizers in the future.
"""

import collections
import copy
import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from corenet.data.datasets.language_modeling import commonsense_170k
from corenet.data.text_tokenizer import build_tokenizer

# Needs to be imported in a special way due to the hyphenated name.
try:
    llmadapters = __import__("LLM-Adapters.commonsense_evaluate")
except:
    llmadapters = None
import argparse
import json
import re
from typing import List, Optional, Tuple

import torch

from corenet.options.opts import get_lm_eval_arguments
from corenet.utils import hf_adapter_utils, logger
from corenet.utils.download_utils import get_local_path

try:
    # For multiple-choice-style evaluation.
    from lm_eval.models.huggingface import HFLM

except:
    HFLM = object

try:
    from transformers import AutoTokenizer, GenerationConfig, LlamaTokenizer
except ModuleNotFoundError as mnfe:
    LlamaTokenizer = None
    AutoTokenizer = None
    GenerationConfig = None


class CoreNetLMEvalWrapper(HFLM):
    """
    A wrapper to build a CoreNet model that inherits the HFLM API.

    Args:
        opts: The global arguments object.
    """

    def __init__(self, opts: argparse.Namespace) -> None:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_config = hf_adapter_utils.CorenetToHFPretrainedConfig(**vars(opts))
        tokenizer_path = getattr(opts, f"text_tokenizer.sentence_piece.model_path")
        tokenizer_path = get_local_path(opts, tokenizer_path)
        # Currently, we only support LLamaTokenizer for this evaluation.
        hf_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        vocab_size = hf_tokenizer.vocab_size
        hf_model = hf_adapter_utils.CorenetToHFPretrainedModel(hf_config, vocab_size)
        hf_model.to(device=device)
        # To ensure padding and batching don't cause issues, for now we only support
        # a batch_size of 1. In some cases, our code handles position ids differently
        # than other codebases, so we are extra cautious.
        batch_size = 1

        super().__init__(
            pretrained=hf_model,
            tokenizer=hf_tokenizer,
            batch_size=batch_size,
            max_length=getattr(opts, "dataset.language_modeling.sequence_length"),
            trust_remote_code=True,
            add_bos_token=getattr(opts, "lm_eval_wrapper.add_sot_token"),
        )

        self.opts = opts


def main_eval_llmadapters(args: Optional[List[str]] = None) -> None:
    """Main entry point for evaluation using LLM Adapters.

    This differs from the lm-eval-harness because it uses a different prompting strategy.
    We currently only support commonsense reasoning tasks.

    Args:
      args: A list of strings, as input on the command line.
    """
    opts = get_lm_eval_arguments(args=args)
    model_eval_wrapper = CoreNetLMEvalWrapper(opts)
    tasks = getattr(opts, "llmadapters_evaluation.datasets")
    dataset_dir = getattr(opts, "llmadapters_evaluation.dataset_dir")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_loc = getattr(opts, "common.results_loc")
    limit = getattr(opts, "llmadapters_evaluation.limit")
    if limit is None:
        limit = int(1e9)
    multiple_choice = getattr(opts, "llmadapters_evaluation.multiple_choice")
    os.makedirs(results_loc, exist_ok=True)

    if multiple_choice:
        main_func = main_multiple_choice
    else:
        main_func = main_generation

    results = {}
    for task in tasks:
        save_file = f"{results_loc}/{task}.json"
        results[task] = main_func(
            task,
            dataset_dir,
            save_file,
            model_eval_wrapper,
            model_eval_wrapper.tokenizer,
            device,
            limit=limit,
        )
    with open(f"{results_loc}/results.json", "w+") as f:
        json.dump(results, f, indent=4)


### Functions for generation-style evaluation.
def main_generation(
    dataset_name: str,
    dataset_dir: str,
    save_file: str,
    model: CoreNetLMEvalWrapper,
    tokenizer: AutoTokenizer,
    device: str,
    limit: int = 1e9,
) -> None:
    """
    Run generation-style evaluation for the given dataset and model.

    Adapted from LLM-Adapters: https://github.com/AGI-Edgerunners/LLM-Adapters

    Args:
        dataset_name: The name of the dataset. E.g. "boolq" or "piqa".
        dataset_dir: The directory which contains a dataset with the given name.
        save_file: The path to which to save the results.
        model: The model to evaluate.
        tokenizer: The tokenizer to use with the model.
        device: The device on which to evaluate.
        limit: The maximum number of samples to process.
    """
    dataset = json.load(open(os.path.join(dataset_dir, dataset_name, "test.json"), "r"))
    # Shuffle the data. This is mainly useful if the ``limit''
    # option is used. This way, results will be sampled from
    # throughout the validation set.
    random.seed(83652)
    random.shuffle(dataset)
    dataset = dataset[:limit]

    total = len(dataset)
    correct = 0
    seen = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, elem in enumerate(dataset):
        seen += 1
        instruction = elem.get("instruction")

        output = evaluate_sample_with_generation(model, tokenizer, instruction, device)

        label = elem.get("answer")
        is_correct = False
        predict = extract_answer_from_generation(dataset_name, output)
        if label == predict:
            correct += 1
            is_correct = True
        new_elem = copy.deepcopy(elem)
        new_elem["output_pred"] = output
        new_elem["pred"] = predict
        new_elem["is_correct"] = is_correct
        output_data.append(new_elem)
        logger.info(f"EXAMPLE:")
        logger.info(f'> INSTRUCTION: {elem["instruction"]}')
        logger.info(f"> OUTPUT: {output}")
        logger.info(f"> PREDICTION: {predict}")
        logger.info(f"> LABEL: {label}")
        logger.info("---------------")
        logger.info(f"\rtest:{idx + 1}/{total} | accuracy: {correct / seen}")
        logger.info("---------------")
        with open(save_file, "w+") as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    logger.info("\n")
    logger.info("Test finished.")
    return {"acc": correct / seen}


def evaluate_sample_with_generation(
    model: CoreNetLMEvalWrapper,
    tokenizer: AutoTokenizer,
    instruction: str,
    device: str,
    temperature: float = 0.1,
    top_p: float = 0.75,
    top_k: int = 40,
    num_beams: int = 4,
    max_new_tokens: int = 32,
    **kwargs: Dict[str, Any],
) -> str:
    """
    Evaluate a sample with the given model.

    Adapted from LLM-Adapters.

    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use with the model.
        instruction: A common sense reasoning input.
        device: The device on which to run.
        temperature: The temperature parameter for generation.
        top_p: The top_p parameter for generation.
        top_k: The top_k parameter for generation.
        num_beams: The number of beams to use in generation.
        max_new_tokens: The maximum number of tokens to generate.
        kwargs: Additional kwargs to pass to the GenerationConfig.
    """
    prompt = llmadapters.commonsense_evaluate.generate_prompt(instruction)
    tokenized_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = tokenized_inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model._model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )
    # The batch size is 1. We remove the batch dimension.
    (s,) = generation_output.sequences
    output = tokenizer.decode(s, skip_special_tokens=True)
    output = output.split("### Response:")[1].strip()
    return output


def extract_answer_from_generation(dataset: str, sentence: str) -> str:
    """
    Extract the multiple-choice answer from a sentence output by the model.

    Adapted from LLM-Adapters: https://github.com/AGI-Edgerunners/LLM-Adapters

    Args:
        dataset: The name of the dataset.
        sentence: The sentence from which to extract the answer.
    Returns:
        A string representing the answer.
    """
    if dataset == "boolq":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"true|false", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "piqa":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"solution1|solution2", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "hellaswag":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == "winogrande":
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"option1|option2", sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        raise ValueError(f"Invalid dataset {dataset}.")


### Functions for MC-style evaluation.
def main_multiple_choice(
    dataset_name: str,
    dataset_dir: str,
    save_file: str,
    model: CoreNetLMEvalWrapper,
    tokenizer: AutoTokenizer,
    device: str,
    limit: int = 1e9,
) -> None:
    """
    Run multiple-choice style evaluation.

    NOTE: The tokenizer and device are unused, but are passed to make
    the API consistent with @main_generation.

    Args:
        dataset_name: The name of the dataset. E.g. "boolq" or "piqa".
        dataset_dir: The directory which contains a dataset with the given name.
        save_file: The path to which to save the results.
        model: The model to evaluate.
        tokenizer: The tokenizer to use with the model.
        device: The device on which to evaluate.
        limit: The maximum number of samples to process.

    """
    dataset = json.load(open(os.path.join(dataset_dir, dataset_name, "test.json"), "r"))
    # Shuffle the data. This is mainly useful if the ``limit''
    # option is used. This way, results will be sampled from
    # throughout the validation set.
    random.seed(83652)
    random.shuffle(dataset)
    dataset = dataset[:limit]

    total = len(dataset)
    correct = 0
    correct_norm = 0
    seen = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, elem in enumerate(dataset):
        seen += 1

        chosen_value, chosen_norm_value = evaluate_likelihood(model, elem)

        label = elem.get("answer")
        is_correct = False
        is_correct_norm = False
        if label == chosen_value:
            correct += 1
            is_correct = True
        if label == chosen_norm_value:
            correct_norm += 1
            is_correct_norm = True
        new_elem = copy.deepcopy(elem)
        new_elem["output_pred"] = chosen_value
        new_elem["output_norm_pred"] = chosen_norm_value
        new_elem["is_correct"] = is_correct
        new_elem["is_correct_norm"] = is_correct_norm
        output_data.append(new_elem)
        logger.info("EXAMPLE:")
        logger.info(f"INSTRUCTION: {elem['instruction']}")
        logger.info(f"OUTPUT: {chosen_value}")
        logger.info(f"OUTPUT NORM: {chosen_norm_value}")
        logger.info(f"LABEL: {label}")
        logger.info("---------------")
        logger.info(
            f"\rtest:{idx + 1}/{total} | accuracy: {correct / seen} | accuracy norm {correct_norm / seen}"
        )
        logger.info("---------------")
        with open(save_file, "w+") as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    logger.info("\n")
    logger.info("Test finished.")
    return {"acc": correct / seen, "acc_norm": correct_norm / seen}


def evaluate_likelihood(
    model: CoreNetLMEvalWrapper, data: Dict[str, str]
) -> Tuple[str, str]:
    """
    Evaluate the likelihood of all possible answers for a prompt. Return
    the most likely one.

    Args:
        model: The model to use for evaluation.
        data: The data point to use. It contains the prompt and all
            possible answers. It contains the following keys: "instruction",
            "input", "output", "answer".
    Returns:
        A tuple containing the strings representing the most likely answers
        as measured by maximum loglikelihood and maximum normalized
        loglikelihood.
    """
    # Generate the set of possible instructions.
    requests = get_loglikelihood_inputs(data)
    model_outputs = model.loglikelihood(requests)

    # Get the log likelihoods and the most likely answer..
    likelihoods = [o[0] for o in model_outputs]
    index = np.argmax(likelihoods)
    output = requests[index].args[1].split(" ")[-1]

    # Get the log likelihoods (normalized by response length)
    # and the most likely answer.
    normalized_likelihoods = [
        likelihoods[i] / len(requests[i].args[1]) for i in range(len(requests))
    ]
    index = np.argmax(normalized_likelihoods)
    output_norm = requests[index].args[1].split(" ")[-1]

    return output, output_norm


Request = collections.namedtuple("Request", "args")


def get_loglikelihood_inputs(data: Dict[str, str]) -> List[Request]:
    """
    Helper function to create the inputs to the model's loglikelihood function.

    Args:
        data: The data point to use. It contains the prompt and all
            possible answers. It contains the following keys: "instruction",
            "input", "output", "answer".

    Returns:
        A list of Requests for the loglikelihood function.
    """

    def generate_response(response_value):
        return f"the correct answer is {response_value}"

    possible_answers = []
    instruction = data["instruction"]
    instruction_end = instruction.split("\n")[-1]
    if re.match(".*Answer format: true/false", instruction_end):
        possible_answers = ["true", "false"]
    else:
        for word in ["answer", "ending", "option", "solution"]:
            if re.match(rf".*Answer format: .*{word}(\d)$", instruction_end):
                num_answers = int(instruction_end[-1])
                possible_answers = [f"{word}{num}" for num in range(1, num_answers + 1)]

    if len(possible_answers) == 0:
        raise ValueError(f"Could not find answer type.")

    context = commonsense_170k.generate_prompt_and_response(
        {"instruction": instruction, "output": "", "input": ""}
    )
    request = []
    for elem in possible_answers:
        obj = Request(((context, generate_response(elem))))
        request.append(obj)
    return request


if __name__ == "__main__":
    main_eval_llmadapters()
