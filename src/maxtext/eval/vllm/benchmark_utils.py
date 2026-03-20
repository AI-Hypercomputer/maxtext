# Taken from vllm-project/tpu-inference (Apache 2.0):
# https://github.com/vllm-project/tpu-inference/blob/main/scripts/vllm/benchmarking/benchmark_utils.py
#
# Copied from vLLM: https://github.com/vllm-project/vllm/blob/02f0c7b/benchmarks/benchmark_utils.py
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This module provides utility functions for benchmarking vLLM.
"""

import argparse
import json
import math
import os
import re
from typing import Any, List, Tuple

import evaluate
import nltk
import numpy as np

from maxtext.eval.vllm.backend_request_func import RequestFuncOutput
from maxtext.eval.vllm.benchmark_dataset import SampleRequest


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record.
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = (extra_info["tensor_parallel_size"])

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(
            records,
            f,
            cls=InfEncoder,
            default=lambda o:
            f"<{type(o).__name__} object is not JSON serializable>",
        )


def postprocess_text_mmlu(preds: List[str],
                          targets: List[str]) -> Tuple[List[int], List[int]]:
    """
    Postprocess the generated text to get the predicted and target answers for the MMLU dataset.
    """
    choices = ["A", "B", "C", "D", None]

    def _parse_answer(output):
        # To match 'assistantfinal' block.
        final_block_match = re.search(r"assistant.*final(.*)", output,
                                      re.IGNORECASE | re.DOTALL)
        if final_block_match:
            final_block = final_block_match.group(1)
            re_str = r"\*\*[^\(]*\s*\(?([A-D])\s*\)?"
            match = re.search(re_str, final_block, re.DOTALL)
            if match:
                return match.group(1).upper()
            re_str = r"(?:choice|answer)[^\(]*\s*\(?([A-D])\s*\)?"
            match = re.search(re_str, final_block, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).upper()

        re_str_fallback = r"(?:thus|so)\s+answer.*(?:option|choice).*\s*\(?([A-D])\s*\)?"
        match = re.search(re_str_fallback, output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()

        re_str_fallback = r"(?:thus|so)\s+answer:?\s*\b([A-D])\b"
        match = re.search(re_str_fallback, output, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()

        re_str_fallback = r"\s*\(([A-D])\)?\s*\w*"
        match = re.search(re_str_fallback, output, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return None

    preds = [choices.index(_parse_answer(pred.strip())) for pred in preds]
    targets = [choices.index(target.strip().upper()) for target in targets]
    return preds, targets


def eval_accuracy_mmlu(request_outputs: List[RequestFuncOutput]) -> dict:
    """
    Evaluate the accuracy of the results of a given benchmark on the MMLU dataset.
    """
    metric = evaluate.load("accuracy")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    preds = []
    targets = []

    for output in request_outputs:
        preds.append(output.generated_text)
        targets.append(output.input_request.completion)
    preds, targets = postprocess_text_mmlu(preds, targets)
    result = metric.compute(
        predictions=preds,
        references=targets,
    )
    result = {k: float(round(np.mean(v), 4)) for k, v in result.items()}
    result["gen_num"] = len(preds)
    return result


def postprocess_text_mlperf(pred: str, target: str):
    """Process a single prediction-target pair for the MLPerf benchmark."""
    pred = pred.strip()
    target = target.strip()
    pred = "\n".join(nltk.sent_tokenize(pred))
    target = "\n".join(nltk.sent_tokenize(target))
    return pred, target


def eval_accuracy_mlperf(request_outputs: List[RequestFuncOutput]) -> dict:
    """
    Evaluate the accuracy of the results of a given benchmark on the MLPerf dataset.
    """
    metric = evaluate.load("rouge")
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    preds = []
    targets = []
    for output in request_outputs:
        pred, target = postprocess_text_mlperf(output.generated_text,
                                               output.input_request.completion)
        preds.append(pred)
        targets.append(target)

    result = metric.compute(
        predictions=preds,
        references=targets,
    )
    result = {k: float(round(np.mean(v) * 100, 4)) for k, v in result.items()}
    result["gen_num"] = len(preds)
    return result


def extract_abcd_gpqa(text: str) -> str:
    """
    Extract answer letter (A, B, C, or D) from GPQA response text.
    Based on gpt-oss abcd_grader.py with patterns for various answer formats.
    """
    patterns = [
        # "Answer: (C)" or "Answers: (B)"
        re.compile(r'(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*\(\s*([ABCD])\s*\)'),
        # "Answer: C" or "Answers – D"
        re.compile(r'(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*([ABCD])\b'),
        # "answer is C" or "answer is (C)"
        re.compile(r'(?ix)\banswer\s+is\s+\(?([ABCD])\)?'),
        # **Answer:** A or *Answers* – B (markdown wrapped)
        re.compile(
            r'''(?ix)(?:\*{1,2}|_{1,2})Answer[s]?\s*[:\-–]?(?:\*{1,2}|_{1,2})\s*([ABCD])\b'''
        ),
        # "Option B" or "Choice: C"
        re.compile(r'(?ix)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCD])\b'),
        # LaTeX \boxed{A}
        re.compile(r'(?x)\\boxed\{[^}]*?([ABCD])[^}]*\}', re.MULTILINE),
        # Bare (A), [B], etc.
        re.compile(
            r'(?x)(?<![A-Za-z0-9])[\(\[]\s*([ABCD])\s*[\)\]](?![A-Za-z0-9])'),
        # Markdown wrapped: *A*, **B**, _C_, __D__
        re.compile(
            r'(?x)(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCD])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])'
        ),
        # Final fallback: line that's exactly "A", "B.", "C)", etc.
        re.compile(
            r'''(?x)^\s*(?:\*{1,2}|_{1,2})?([ABCD])(?:\*{1,2}|_{1,2})?\s*[\.\)\-–:]?\s*.*$''',
            re.MULTILINE),
    ]

    # Also check for gpt-oss style "assistantfinal" block
    final_block_match = re.search(r"assistant.*final(.*)", text,
                                  re.IGNORECASE | re.DOTALL)
    if final_block_match:
        final_block = final_block_match.group(1)
        match = re.search(r"\*\*[^\(]*\s*\(?([A-D])\s*\)?", final_block,
                          re.DOTALL)
        if match:
            return match.group(1).upper()
        match = re.search(r"(?:choice|answer)[^\(]*\s*\(?([A-D])\s*\)?",
                          final_block, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).upper()

    for pat in patterns:
        m = pat.search(text)
        if m:
            letter = m.group(1).upper()
            if letter in 'ABCD':
                return letter

    first_char = text.strip()[:1].upper()
    if first_char in 'ABCD':
        return first_char

    return None


def eval_accuracy_gpqa(request_outputs: List[RequestFuncOutput]) -> dict:
    """
    Evaluate the accuracy of the results on the GPQA dataset.
    """
    correct = 0
    total = 0

    for output in request_outputs:
        if not output.success:
            continue

        generated_text = output.generated_text
        target = output.input_request.completion  # 'A', 'B', 'C', or 'D'

        extracted = extract_abcd_gpqa(generated_text)
        if extracted == target.upper():
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    result = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "gen_num": len(request_outputs),
    }
    return result


def eval_benchmark_dataset_result(request_outputs: List[RequestFuncOutput],
                                  dataset_name: str) -> dict:
    """
    Evaluate the accuracy of the results of a given benchmark on a given dataset.
    """
    if dataset_name == "mmlu":
        return eval_accuracy_mmlu(request_outputs)
    elif dataset_name in ("mlperf", "mlperf_openorca", "openorca"):
        return eval_accuracy_mlperf(request_outputs)
    elif dataset_name in ("gpqa", "gpqa_diamond"):
        return eval_accuracy_gpqa(request_outputs)
    else:
        raise NotImplementedError("Evaluation is not supported for dataset: %s" %
                                  dataset_name)


def sample_warmup_requests(requests: List[SampleRequest]):
    """
    Sample warmup requests from a list of requests by prompt-length bucket.
    """
    interesting_buckets = [0, 16, 32, 64, 128, 256, 512, 1024]

    for start, end in zip(interesting_buckets[:-1], interesting_buckets[1:]):
        for request in requests:
            if start < request.prompt_len <= end:
                yield request
                break
