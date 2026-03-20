# Taken from vllm-project/tpu-inference (Apache 2.0):
# https://github.com/vllm-project/tpu-inference/blob/main/scripts/vllm/benchmarking/benchmark_dataset.py
#
# Copied from vLLM: https://github.com/vllm-project/vllm/blob/02f0c7b/benchmarks/benchmark_dataset.py
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# MaxText note: vLLM-specific imports (LoRARequest, MultiModalDataDict) are
# made optional so this file can be imported without a vLLM installation.
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - MMLMDataset
  - MLPerfDataset
  - Random (synthetic)
  - Sonnet
"""

import logging
import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerBase

# vLLM-specific types — optional; only needed when running inside a vLLM env.
try:
    from vllm.lora.request import LoRARequest
    from vllm.multimodal import MultiModalDataDict
except ImportError:
    LoRARequest = None  # type: ignore[assignment,misc]
    MultiModalDataDict = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class SampleRequest:
    """
    Represents a single inference request for benchmarking.
    """

    prompt: Union[str, Any]
    prompt_len: int
    expected_output_len: int
    multi_modal_data: Optional[Union[Any, dict, list[dict]]] = None
    lora_request: Optional[Any] = None
    completion: Optional[str] = None
    request_id: Optional[str] = None


# -----------------------------------------------------------------------------
# Benchmark Dataset Base Class
# -----------------------------------------------------------------------------


class BenchmarkDataset(ABC):
    DEFAULT_SEED = 0
    IS_MULTIMODAL = False

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        random_seed: int = DEFAULT_SEED,
    ) -> None:
        self.dataset_path = dataset_path
        self.random_seed = random_seed if random_seed is not None else self.DEFAULT_SEED
        self.data = None

    def apply_multimodal_chat_transformation(
            self,
            prompt: str,
            mm_content=None) -> list[dict]:
        content = [{"text": prompt, "type": "text"}]
        if mm_content is not None:
            content.append(mm_content)
        return [{"role": "user", "content": content}]

    def load_data(self) -> None:
        raise NotImplementedError(
            "load_data must be implemented in subclasses.")

    @abstractmethod
    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        request_id_prefix: str = "",
    ) -> list[SampleRequest]:
        raise NotImplementedError("sample must be implemented in subclasses.")

    def maybe_oversample_requests(
        self,
        requests: list[SampleRequest],
        num_requests: int,
        request_id_prefix: str = "",
    ) -> None:
        if len(requests) < num_requests:
            random.seed(self.random_seed)
            additional = deepcopy(
                random.choices(requests, k=num_requests - len(requests)))
            for i in range(len(additional)):
                req = additional[i]
                req.request_id = request_id_prefix + str(len(requests) + i)
            requests.extend(additional)
            logger.info("Oversampled requests to reach %d total samples.",
                        num_requests)


# -----------------------------------------------------------------------------
# Utility Functions and Global Caches
# -----------------------------------------------------------------------------


def is_valid_sequence(
    prompt_len: int,
    output_len: int,
    min_len: int = 4,
    max_prompt_len: int = 1024,
    max_total_len: int = 2048,
    skip_min_output_len_check: bool = False,
) -> bool:
    prompt_too_short = prompt_len < min_len
    output_too_short = (not skip_min_output_len_check) and (output_len < min_len)
    prompt_too_long = prompt_len > max_prompt_len
    combined_too_long = (prompt_len + output_len) > max_total_len
    return not (prompt_too_short or output_too_short or prompt_too_long
                or combined_too_long)


# -----------------------------------------------------------------------------
# MMLU Dataset Implementation
# -----------------------------------------------------------------------------


class MMLUDataset(BenchmarkDataset):
    """
    Implements the MMLUDataset dataset.  Logic heavily inspired by Jetstream
    https://github.com/AI-Hypercomputer/JetStream/blob/bbfb5bd/benchmarks/benchmark_serving.py#L327.
    """

    def __init__(self, num_shots: int, mmlu_method: str,
                 use_chat_template: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mmlu_method = mmlu_method
        self.num_shots = num_shots
        self.use_chat_template = use_chat_template
        self.load_data()

    def load_mmlu_dataset_csv(self,
                              dataset_path: str) -> tuple[Any, dict[str, str]]:
        assert dataset_path != ""
        dataset = []
        prompts_per_subject = dict()
        for cvs_file in os.listdir(dataset_path):
            if cvs_file.endswith(".csv"):
                subject = " ".join(cvs_file.split("_")[:-1])
                if subject not in prompts_per_subject:
                    prompts_per_subject[subject] = ""
                filepath = os.path.join(dataset_path, cvs_file)
                data = pd.read_csv(filepath, header=None)
                data["subject"] = subject
                dataset.append(data)

        combined_dataset = pd.concat(dataset, ignore_index=True)
        header_dict = {
            0: "question",
            1: "A",
            2: "B",
            3: "C",
            4: "D",
            5: "answer",
        }
        combined_dataset.rename(columns=header_dict, inplace=True)
        return combined_dataset, prompts_per_subject

    def gen_mmlu_qa(self, data: Any, mmlu_method: str = "") -> str:
        output = ""
        for _, row in data.iterrows():
            output += (f"\nQuestion: {row['question']}\n"
                       f"Choices:\n"
                       f"(A) {row['A']}\n"
                       f"(B) {row['B']}\n"
                       f"(C) {row['C']}\n"
                       f"(D) {row['D']}\n")
            if self.use_chat_template:
                output += "\nPlease think carefully and give the answer. "
                if mmlu_method == "HELM":
                    output += f"**Answer: ({row['answer']})** <|endoftext|>\n"
                elif mmlu_method == "Harness":
                    content = row[row["answer"].upper()]
                    output += f"**Answer: ({row['answer']}) {content}\n\n"
            else:
                output += "\nCorrect answer:"
                if mmlu_method == "HELM":
                    output += f"({row['answer']})\n\n"
                elif mmlu_method == "Harness":
                    content = row[row["answer"].upper()]
                    output += f"({row['answer']}) {content}\n\n"
        return output

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for loading data.")

        combined_dataset, prompts_per_subject = self.load_mmlu_dataset_csv(
            self.dataset_path)
        num_rows, _ = combined_dataset.shape
        print(f"Loaded {num_rows} data from mmlu dataset")

        for subject in prompts_per_subject:
            header = (
                f"The following are multiple choice questions (with answers) "
                f"about {subject}:\n")
            shots_data = combined_dataset[combined_dataset["subject"] ==
                                          subject].head(self.num_shots)
            prompts_per_subject[subject] = header + self.gen_mmlu_qa(
                shots_data, mmlu_method=self.mmlu_method)

        mmlu_data = []
        for _, row in combined_dataset.iloc[self.num_shots:].iterrows():
            question_prompt = self.gen_mmlu_qa(pd.DataFrame([row]))
            output = row["answer"]
            prompt = prompts_per_subject[row["subject"]] + question_prompt
            mmlu_data.append((prompt, output))

        self.data = mmlu_data

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
        enable_multimodal_chat: bool = False,
        **kwargs,
    ) -> list:
        samples: list = []
        for prompt, completion in self.data:
            if len(samples) >= num_requests:
                break

            if self.use_chat_template:
                messages = [{
                    "role": "system",
                    "content": "Reasoning effort: high"
                }, {
                    "role": "user",
                    "content": prompt
                }]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                except Exception as e:
                    logger.error(f"Could not apply chat template: {e}. "
                                 "Falling back to raw prompt.")

            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = len(
                completion_ids) if output_len is None else output_len
            if input_len is not None and input_len <= prompt_len:
                raise ValueError(
                    f"prompt is too short: prompt_len is {prompt_len} but input_len is {input_len}"
                )
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len or new_output_len,
                    completion=completion,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# MLPerf Dataset Implementation
# -----------------------------------------------------------------------------


class MLPerfDataset(BenchmarkDataset):
    """Implements the MLPerf dataset."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        dataset = pd.read_pickle(self.dataset_path)
        mlperf_data = []
        print(f"Loaded {len(dataset)} data from mlperf dataset")
        for _, row in dataset.iterrows():
            prompt = row["question"]
            output = row["output"]
            mlperf_data.append((prompt, output))
        self.data = mlperf_data

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list:
        samples: list = []
        for prompt, completion in self.data:
            if len(samples) >= num_requests:
                break
            prompt_ids = tokenizer(prompt).input_ids
            completion_ids = tokenizer(completion).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = len(
                completion_ids) if output_len is None else output_len
            if input_len is not None and input_len <= prompt_len:
                raise ValueError(
                    f"prompt is too short: prompt_len is {prompt_len} but input_len is {input_len}"
                )
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len or new_output_len,
                    completion=completion,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# GPQA Dataset Implementation
# -----------------------------------------------------------------------------


class GPQADataset(BenchmarkDataset):
    """Implements the GPQA dataset. Uses the diamond variant."""

    QUERY_TEMPLATE = """{Question}

(A) {A}
(B) {B}
(C) {C}
(D) {D}

Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'."""

    def __init__(self, use_chat_template: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_chat_template = use_chat_template
        self.load_data()

    def load_data(self) -> None:
        if self.dataset_path:
            df = pd.read_csv(self.dataset_path)
        else:
            url = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
            df = pd.read_csv(url)

        rng = random.Random(self.random_seed)
        examples = [row.to_dict() for _, row in df.iterrows()]
        examples = [
            example | {"permutation": rng.sample(range(4), 4)}
            for example in examples
        ]

        gpqa_data = []
        for row in examples:
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(A=choices[0], B=choices[1], C=choices[2],
                                D=choices[3], Question=row["Question"])
            prompt = self.QUERY_TEMPLATE.format(**choices_dict)
            gpqa_data.append((prompt, correct_answer))

        self.data = gpqa_data
        print(f"Loaded {len(self.data)} examples from GPQA diamond dataset")

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        input_len: Optional[int] = None,
        output_len: Optional[int] = None,
        **kwargs,
    ) -> list:
        samples: list = []
        for prompt, completion in self.data:
            if len(samples) >= num_requests:
                break
            if self.use_chat_template:
                messages = [{"role": "system", "content": "Reasoning effort: high"},
                            {"role": "user", "content": prompt}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                except Exception as e:
                    logger.error(f"Could not apply chat template: {e}. "
                                 "Falling back to raw prompt.")
            prompt_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_ids)
            new_output_len = output_len if output_len is not None else 2048
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=new_output_len,
                    completion=completion,
                ))
        self.maybe_oversample_requests(samples, num_requests)
        return samples


# -----------------------------------------------------------------------------
# Random Dataset Implementation (Synthetic Data)
# -----------------------------------------------------------------------------


class RandomDataset(BenchmarkDataset):
    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list[SampleRequest]:
        assert range_ratio < 1.0
        vocab_size = tokenizer.vocab_size
        num_special_tokens = tokenizer.num_special_tokens_to_add()
        real_input_len = input_len - num_special_tokens
        prefix_token_ids = (np.random.randint(
            0, vocab_size, size=prefix_len).tolist() if prefix_len > 0 else [])
        input_low = int(real_input_len * (1 - range_ratio))
        input_high = int(real_input_len * (1 + range_ratio))
        output_low = max(int(output_len * (1 - range_ratio)), 1)
        output_high = int(output_len * (1 + range_ratio))
        input_lens = np.random.randint(input_low, input_high + 1, size=num_requests)
        output_lens = np.random.randint(output_low, output_high + 1, size=num_requests)
        offsets = np.random.randint(0, vocab_size, size=num_requests)
        requests = []
        for i in range(num_requests):
            inner_seq = ((offsets[i] + i + np.arange(input_lens[i])) % vocab_size).tolist()
            token_sequence = prefix_token_ids + inner_seq
            prompt = tokenizer.decode(token_sequence)
            total_input_len = prefix_len + int(input_lens[i])
            re_encoded_sequence = tokenizer.encode(
                prompt, add_special_tokens=False)[:total_input_len]
            prompt = tokenizer.decode(re_encoded_sequence)
            total_input_len = len(re_encoded_sequence)
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=total_input_len,
                    expected_output_len=int(output_lens[i]),
                    request_id=request_id_prefix + str(i),
                ))
        return requests


# -----------------------------------------------------------------------------
# Sonnet Dataset Implementation
# -----------------------------------------------------------------------------


class SonnetDataset(BenchmarkDataset):
    DEFAULT_PREFIX_LEN = 200
    DEFAULT_INPUT_LEN = 550
    DEFAULT_OUTPUT_LEN = 150

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        if not self.dataset_path:
            raise ValueError("dataset_path must be provided.")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = f.readlines()

    def sample(
        self,
        tokenizer,
        num_requests: int,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        return_prompt_formatted: bool = False,
        request_id_prefix: str = "",
        **kwargs,
    ) -> list:
        tokenized_lines = [tokenizer(line).input_ids for line in self.data]
        avg_len = sum(len(tokens) for tokens in tokenized_lines) / len(tokenized_lines)
        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        base_msg = [{"role": "user", "content": base_prompt}]
        base_fmt = tokenizer.apply_chat_template(base_msg, add_generation_prompt=True, tokenize=False)
        base_offset = len(tokenizer(base_fmt).input_ids)
        if input_len <= base_offset:
            raise ValueError(f"'input_len' must be higher than the base prompt length ({base_offset}).")
        num_input_lines = round((input_len - base_offset) / avg_len)
        num_prefix_lines = max(round((prefix_len - base_offset) / avg_len), 0)
        prefix_lines = self.data[:num_prefix_lines]
        samples = []
        ind = 0
        while len(samples) < num_requests:
            extra_lines = random.choices(self.data, k=num_input_lines - num_prefix_lines)
            prompt = f"{base_prompt}{''.join(prefix_lines + extra_lines)}"
            msg = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
            prompt_len = len(tokenizer(prompt_formatted).input_ids)
            if prompt_len <= input_len:
                samples.append(
                    SampleRequest(
                        prompt=prompt_formatted if return_prompt_formatted else prompt,
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                        request_id=request_id_prefix + str(ind),
                    ))
                ind += 1
        return samples
