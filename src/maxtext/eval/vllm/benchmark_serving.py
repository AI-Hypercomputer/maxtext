# Taken from vllm-project/tpu-inference (Apache 2.0):
# https://github.com/vllm-project/tpu-inference/blob/main/scripts/vllm/benchmarking/benchmark_serving.py
#
# Copied from vLLM: https://github.com/vllm-project/vllm/blob/02f0c7b/benchmarks/benchmark_serving.py
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests

On the client side, run:
    python -m eval.vllm.benchmark_serving \
        --backend vllm \
        --model <your_model> \
        --dataset-name mmlu \
        --dataset-path <path to dataset> \
        --num-prompts 1000
"""

import argparse
import asyncio
import gc
import random
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass
from typing import Optional

import numpy as np

from maxtext.eval.vllm.backend_request_func import (ASYNC_REQUEST_FUNCS,
                                               OPENAI_COMPATIBLE_BACKENDS,
                                               RequestFuncInput,
                                               RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from maxtext.eval.vllm.backend_request_func import get_tokenizer

try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from maxtext.eval.vllm.benchmark_dataset import (GPQADataset, MLPerfDataset,
                                            MMLUDataset, RandomDataset,
                                            SampleRequest, SonnetDataset)
from maxtext.eval.vllm.benchmark_utils import (eval_benchmark_dataset_result,
                                          sample_warmup_requests)

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[SampleRequest, None]:
    input_requests: Iterable[SampleRequest] = iter(input_requests)
    assert burstiness > 0
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request
        if request_rate == float("inf"):
            continue
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens
            if not output_len:
                output_len = len(
                    tokenizer(outputs[i].generated_text,
                              add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []
        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(goodput_config_dict["ttft"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(goodput_config_dict["tpot"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(goodput_config_dict["e2el"] /
                              MILLISECONDS_TO_SECONDS_CONVERSION)
        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000)
                             for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000)
                            for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000)
                             for p in selected_percentiles],
    )
    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: list[SampleRequest],
    logprobs: Optional[int],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentiles: list[float],
    ignore_eos: bool,
    goodput_config_dict: dict[str, float],
    max_concurrency: Optional[int],
    extra_body: Optional[dict],
    warmup_mode: str = "sampled",
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    warmup_requests = None
    if warmup_mode == "full":
        warmup_requests = input_requests
    elif warmup_mode == "sampled":
        warmup_requests = list(sample_warmup_requests(input_requests)) * 2

    test_prompt = test_prompt_len = test_output_len = test_mm_content = None
    if warmup_requests:
        print(f"Warmup (mode: {warmup_mode}) is starting.")
        for warmup_request in tqdm(warmup_requests):
            test_prompt, test_prompt_len, test_output_len, test_mm_content = (
                warmup_request.prompt,
                warmup_request.prompt_len,
                warmup_request.expected_output_len,
                warmup_request.multi_modal_data,
            )
            assert test_mm_content is None or isinstance(test_mm_content, dict)
            test_input = RequestFuncInput(
                model=model_id,
                model_name=model_name,
                prompt=test_prompt,
                api_url=api_url,
                prompt_len=test_prompt_len,
                output_len=test_output_len,
                logprobs=logprobs,
                multi_modal_content=test_mm_content,
                ignore_eos=ignore_eos,
                extra_body=extra_body,
            )
            test_output = await request_func(request_func_input=test_input)
            if not test_output.success:
                raise ValueError(
                    "Warmup failed - Please make sure benchmark arguments "
                    f"are correctly specified. Error: {test_output.error}")
        print(f"Warmup (mode: {warmup_mode}) has completed.")

    if profile:
        print("Starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
            multi_modal_content=test_mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate, burstiness):
        prompt, prompt_len, output_len, mm_content = (
            request.prompt,
            request.prompt_len,
            request.expected_output_len,
            request.multi_modal_data,
        )
        req_model_id, req_model_name = model_id, model_name
        request_kwargs = dict(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            logprobs=logprobs,
            multi_modal_content=mm_content,
            ignore_eos=ignore_eos,
            extra_body=extra_body,
        )
        if request.completion is not None:
            request_kwargs["completion"] = request.completion
        if request.request_id is not None:
            request_kwargs["request_id"] = request.request_id
        request_func_input = RequestFuncInput(**request_kwargs)
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)))
    outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("Stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",
            prompt_len=test_prompt_len,
            output_len=test_output_len,
            logprobs=logprobs,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if goodput_config_dict:
        print("{:<40} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total token throughput (tok/s):", metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }

    def process_one_metric(metric_attribute_name, metric_name, metric_header):
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print("{:<40} {:<10.2f}".format(
            f"Mean {metric_name} (ms):",
            getattr(metrics, f"mean_{metric_attribute_name}_ms"),
        ))
        print("{:<40} {:<10.2f}".format(
            f"Median {metric_name} (ms):",
            getattr(metrics, f"median_{metric_attribute_name}_ms"),
        ))
        result[f"mean_{metric_attribute_name}_ms"] = getattr(metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(metrics, f"std_{metric_attribute_name}_ms")
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")
    print("=" * 50)

    return result, outputs


def check_goodput_args(args):
    goodput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(f"Invalid metric name found, {slo_name}: {slo_val}.")
            if slo_val < 0:
                raise ValueError(f"Invalid value found, {slo_name}: {slo_val}.")
    return goodput_config_dict


def parse_goodput(slo_pairs):
    goodput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives.") from err
    return goodput_config_dict


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    tokenizer_mode = args.tokenizer_mode

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id, tokenizer_mode=tokenizer_mode,
                              trust_remote_code=args.trust_remote_code)

    if args.dataset_name is None:
        raise ValueError("Please specify '--dataset-name'.")

    if args.dataset_name == "sonnet":
        dataset = SonnetDataset(dataset_path=args.dataset_path)
        if args.backend == "openai-chat":
            input_requests = dataset.sample(
                num_requests=args.num_prompts, input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len, prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer, return_prompt_formatted=False,
                request_id_prefix=args.request_id_prefix)
        else:
            assert tokenizer.chat_template or tokenizer.default_chat_template
            input_requests = dataset.sample(
                num_requests=args.num_prompts, input_len=args.sonnet_input_len,
                output_len=args.sonnet_output_len, prefix_len=args.sonnet_prefix_len,
                tokenizer=tokenizer, return_prompt_formatted=True,
                request_id_prefix=args.request_id_prefix)
    else:
        dataset_mapping = {
            "mmlu": lambda: MMLUDataset(
                random_seed=args.seed, dataset_path=args.dataset_path,
                num_shots=args.mmlu_num_shots, mmlu_method=args.mmlu_method,
                use_chat_template=args.mmlu_use_chat_template).sample(
                    tokenizer=tokenizer, num_requests=args.num_prompts,
                    input_len=args.mmlu_input_len, output_len=args.mmlu_output_len),
            "mlperf": lambda: MLPerfDataset(
                random_seed=args.seed, dataset_path=args.dataset_path).sample(
                    tokenizer=tokenizer, num_requests=args.num_prompts,
                    input_len=args.mlperf_input_len, output_len=args.mlperf_output_len),
            "gpqa": lambda: GPQADataset(
                random_seed=args.seed, dataset_path=args.dataset_path,
                use_chat_template=args.gpqa_use_chat_template).sample(
                    tokenizer=tokenizer, num_requests=args.num_prompts,
                    output_len=args.gpqa_output_len),
            "random": lambda: RandomDataset(
                random_seed=args.seed, dataset_path=args.dataset_path).sample(
                    tokenizer=tokenizer, num_requests=args.num_prompts,
                    prefix_len=args.random_prefix_len, input_len=args.random_input_len,
                    output_len=args.random_output_len, range_ratio=args.random_range_ratio,
                    request_id_prefix=args.request_id_prefix),
        }
        try:
            input_requests = dataset_mapping[args.dataset_name]()
        except KeyError as err:
            raise ValueError(f"Unknown dataset: {args.dataset_name}") from err

    goodput_config_dict = check_goodput_args(args)
    sampling_params = {
        k: v for k, v in {
            "top_p": args.top_p, "top_k": args.top_k,
            "min_p": args.min_p, "temperature": args.temperature,
        }.items() if v is not None
    }
    if sampling_params and args.backend not in OPENAI_COMPATIBLE_BACKENDS:
        raise ValueError("Sampling parameters are only supported by openai-compatible backends.")
    if "temperature" not in sampling_params:
        sampling_params["temperature"] = 0.0
    if args.backend == "llama.cpp":
        sampling_params["cache_prompt"] = False

    gc.collect()
    gc.freeze()

    print("Using sampling parameters:", sampling_params)
    _, request_outputs = asyncio.run(
        benchmark(
            backend=backend, api_url=api_url, base_url=base_url,
            model_id=model_id, model_name=model_name, tokenizer=tokenizer,
            input_requests=input_requests, logprobs=args.logprobs,
            request_rate=args.request_rate, burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm, profile=args.profile,
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos, goodput_config_dict=goodput_config_dict,
            max_concurrency=args.max_concurrency, extra_body=sampling_params,
            warmup_mode=args.warmup_mode,
        ))

    if args.run_eval:
        eval_benchmark_dataset_result(request_outputs, args.dataset_name)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=list(ASYNC_REQUEST_FUNCS.keys()))
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/v1/completions")
    parser.add_argument("--dataset-name", type=str, default="sharegpt",
                        choices=["sharegpt", "burstgpt", "sonnet", "random",
                                 "hf", "custom", "mmlu", "mlperf", "gpqa"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--percentile-metrics", type=str, default="ttft,tpot,itl")
    parser.add_argument("--metric-percentiles", type=str, default="99")
    parser.add_argument("--goodput", nargs="+", required=False)
    parser.add_argument("--request-id-prefix", type=str, default="benchmark-serving")

    mmlu_group = parser.add_argument_group("mmlu dataset options")
    mmlu_group.add_argument("--mmlu-input-len", type=int, default=None)
    mmlu_group.add_argument("--mmlu-output-len", type=int, default=None)
    mmlu_group.add_argument("--mmlu-num-shots", type=int, default=1)
    mmlu_group.add_argument("--mmlu-method", type=str, default="HELM",
                            choices=["HELM", "Harness", ""])
    mmlu_group.add_argument("--mmlu-use-chat-template", action="store_true")

    mlperf_group = parser.add_argument_group("mlperf dataset options")
    mlperf_group.add_argument("--mlperf-input-len", type=int, default=None)
    mlperf_group.add_argument("--mlperf-output-len", type=int, default=None)

    gpqa_group = parser.add_argument_group("gpqa dataset options")
    gpqa_group.add_argument("--gpqa-output-len", type=int, default=2048)
    gpqa_group.add_argument("--gpqa-use-chat-template", action="store_true")

    sonnet_group = parser.add_argument_group("sonnet dataset options")
    sonnet_group.add_argument("--sonnet-input-len", type=int, default=550)
    sonnet_group.add_argument("--sonnet-output-len", type=int, default=150)
    sonnet_group.add_argument("--sonnet-prefix-len", type=int, default=200)

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument("--random-input-len", type=int, default=1024)
    random_group.add_argument("--random-output-len", type=int, default=128)
    random_group.add_argument("--random-range-ratio", type=float, default=0.0)
    random_group.add_argument("--random-prefix-len", type=int, default=0)

    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--temperature", type=float, default=None)

    parser.add_argument("--tokenizer-mode", type=str, default="auto",
                        choices=["auto", "slow", "mistral", "custom"])
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--warmup-mode", type=str, default="sampled",
                        choices=["none", "sampled", "full"])

    args = parser.parse_args()
    main(args)
