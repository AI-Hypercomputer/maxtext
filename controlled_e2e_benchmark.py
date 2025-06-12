#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Controlled End-to-End Benchmark for MaxText
This benchmark bridges the gap between microbenchmarks and full MLPerf runs.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import argparse
import random
import queue
import threading
import json
import os
import io
import re
import contextlib
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

from MaxText import max_utils
from MaxText import maxengine
from MaxText import pyconfig


@dataclass
class Request:
    """Represents a single inference request"""
    id: int
    prefill_len: int
    decode_len: int
    arrival_time: float
    tokens: Optional[jax.Array] = None
    true_length: Optional[int] = None

    # Timing fields
    prefill_start_time: Optional[float] = None
    prefill_end_time: Optional[float] = None
    first_token_time: Optional[float] = None
    decode_start_time: Optional[float] = None
    decode_end_time: Optional[float] = None
    completion_time: Optional[float] = None

    # Results
    output_tokens: List[int] = field(default_factory=list)
    slot_used: Optional[int] = None


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics"""
    total_requests: int = 0
    completed_requests: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    max_concurrent_tokens: int = 0

    # Timing
    benchmark_start_time: float = 0.0
    benchmark_end_time: float = 0.0

    # Latency tracking
    prefill_latencies: List[float] = field(default_factory=list)
    decode_latencies: List[float] = field(default_factory=list)
    time_to_first_token: List[float] = field(default_factory=list)
    request_latencies: List[float] = field(default_factory=list)
    decode_step_latencies: List[float] = field(default_factory=list)

    # Memory tracking
    mem_after_load_gb: Optional[float] = None
    mem_after_init_gb: Optional[float] = None

    # Throughput tracking
    tokens_per_second_history: List[Tuple[float, float]] = field(default_factory=list)

    # Slot utilization
    slot_utilization_history: List[Tuple[float, float]] = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        total_time = self.benchmark_end_time - self.benchmark_start_time
        total_tokens = self.total_prefill_tokens + self.total_decode_tokens

        def safe_percentile(data, p):
            return np.percentile(data, p) if data else 0.0

        summary = {
            "total_requests": self.total_requests,
            "completed_requests": self.completed_requests,
            "total_time_s": total_time,
            "total_tokens": total_tokens,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decode_tokens": self.total_decode_tokens,
            "throughput_tokens_per_s": total_tokens / total_time if total_time > 0 else 0,
            "prefill_throughput_tokens_per_s": self.total_prefill_tokens / total_time if total_time > 0 else 0,
            "decode_throughput_tokens_per_s": self.total_decode_tokens / total_time if total_time > 0 else 0,
            "avg_prefill_latency_ms": np.mean(self.prefill_latencies) * 1000 if self.prefill_latencies else 0,
            "p50_prefill_latency_ms": safe_percentile(self.prefill_latencies, 50) * 1000,
            "p99_prefill_latency_ms": safe_percentile(self.prefill_latencies, 99) * 1000,
            "avg_decode_latency_s": np.mean(self.decode_latencies) if self.decode_latencies else 0,
            "p50_decode_latency_s": safe_percentile(self.decode_latencies, 50),
            "p99_decode_latency_s": safe_percentile(self.decode_latencies, 99),
            "avg_time_to_first_token_ms": np.mean(self.time_to_first_token) * 1000 if self.time_to_first_token else 0,
            "p50_time_to_first_token_ms": safe_percentile(self.time_to_first_token, 50) * 1000,
            "p99_time_to_first_token_ms": safe_percentile(self.time_to_first_token, 99) * 1000,
            "avg_request_latency_s": np.mean(self.request_latencies) if self.request_latencies else 0,
            "p50_request_latency_s": safe_percentile(self.request_latencies, 50),
            "p99_request_latency_s": safe_percentile(self.request_latencies, 99),
            "avg_time_per_decode_step_ms": np.mean(self.decode_step_latencies) * 1000 if self.decode_step_latencies else 0,
            "p50_time_per_decode_step_ms": safe_percentile(self.decode_step_latencies, 50) * 1000,
            "p99_time_per_decode_step_ms": safe_percentile(self.decode_step_latencies, 99) * 1000,
        }
        return summary


class ControlledBenchmark:
    def __init__(self, engine: maxengine.MaxEngine, config: Any, benchmark_config: Dict[str, Any]):
        self.engine = engine
        self.config = config
        self.benchmark_config = benchmark_config
        self.params = None
        self.decode_state = None
        self.request_queue = queue.Queue()
        self.active_slots: Dict[int, Request] = {}
        self.completed_requests: List[Request] = []
        self.free_slots = list(range(engine.max_concurrent_decodes))
        self.metrics = BenchmarkMetrics()
        self.is_running = False
        self.all_requests_submitted = False
        self.decode_thread = None
        self.metrics_thread = None
        self.last_decode_time = None
        self.decode_step_count = 0

    def _log_and_capture_memory_usage(self, stage: str) -> Optional[float]:
        """Helper to print memory stats and return the usage for the first TPU device."""
        separator = "=" * 25
        print(f"\n{separator} MEMORY USAGE: {stage.upper()} {separator}")

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            max_utils.print_mem_stats(stage)
        captured_output = s.getvalue()
        
        print(captured_output, end="")
        print("=" * (52 + len(stage)) + "\n")

        try:
            match = re.search(r"Using \(GB\) ([\d.]+)", captured_output)
            if match:
                return float(match.group(1))
        except (ValueError, IndexError):
            return None
        return None

    def load_and_warmup(self):
        """Load parameters, capture memory usage, and perform warmup."""
        print("Loading parameters...")
        self.params = self.engine.load_params()
        self.metrics.mem_after_load_gb = self._log_and_capture_memory_usage("After Loading Parameters")

        print("Initializing decode state...")
        self.decode_state = self.engine.init_decode_state()
        self.metrics.mem_after_init_gb = self._log_and_capture_memory_usage("After Initializing Decode State")

        if self.benchmark_config.get('warmup_steps', 0) > 0:
            print(f"Running {self.benchmark_config['warmup_steps']} warmup steps...")
            self._run_warmup()

    def _run_warmup(self):
        """Run warmup iterations"""
        warmup_steps = self.benchmark_config.get('warmup_steps', 0)
        warmup_batch_size = min(8, self.engine.max_concurrent_decodes)
        if warmup_batch_size == 0:
            print("Skipping warmup: max_concurrent_decodes is 0.")
            return

        for i in range(warmup_batch_size):
            dummy_tokens = jnp.zeros((128,), dtype=jnp.int32)
            prefix, _ = self.engine.prefill(
                params=self.params,
                padded_tokens=dummy_tokens,
                true_length=128,
                slot=i
            )
            self.decode_state = self.engine.insert(
                prefix=prefix,
                decode_state=self.decode_state,
                slot=i
            )

        for _ in range(warmup_steps):
            self.decode_state, _ = self.engine.generate(
                params=self.params,
                decode_state=self.decode_state
            )

        for i in range(warmup_batch_size):
            self.engine.free_resource(i)

        print("Warmup complete")

    def generate_workload(self, num_requests: int, seed: int) -> List[Request]:
        """Generate workload with new, more specific and corrected workload types."""
        workload_type = self.benchmark_config.get('workload_type', 'realistic')
        print(f"Generating workload with {num_requests} requests of type '{workload_type}'...")
        rng = random.Random(seed)
        max_prefill = self.config.max_prefill_predict_length
        max_target = self.config.max_target_length

        requests = []
        for i in range(num_requests):
            prefill_len = 0
            decode_len = 0
            
            # --- REVISED AND CORRECTED WORKLOAD DEFINITIONS ---

            if workload_type == 'chat':
                # Simulates interactive, conversational AI.
                # Many short prompts, and relatively short, conversational responses.
                prefill_len = rng.randint(16, 256)
                max_decode_for_this_req = max_target - prefill_len
                decode_len = rng.randint(20, min(256, max_decode_for_this_req))

            elif workload_type == 'summarize':
                # CORRECTED: Simulates summarizing a document.
                # Long prompt (the document), short-to-medium response (the summary).
                prefill_len = rng.randint(max_prefill // 2, max_prefill)
                max_decode_for_this_req = max_target - prefill_len
                # Summary is typically a fraction of the original length, e.g., 10-25%
                decode_len = rng.randint(max(10, int(prefill_len * 0.1)), min(int(prefill_len * 0.25), max_decode_for_this_req))

            elif workload_type == 'rag':
                # Simulates Retrieval-Augmented Generation.
                # Long prompt (user query + retrieved context), medium response (synthesized answer).
                # Similar to summarize but with potentially shorter responses.
                prefill_len = rng.randint(max_prefill // 2, max_prefill)
                max_decode_for_this_req = max_target - prefill_len
                decode_len = rng.randint(100, min(512, max_decode_for_this_req))

            elif workload_type == 'generative':
                # Simulates creative tasks like story writing or code generation.
                # Short prompt ("write a poem about..."), very long response.
                prefill_len = rng.randint(16, 128)
                max_decode_for_this_req = max_target - prefill_len
                # Generate a response that is a substantial portion of the remaining length
                decode_len = rng.randint(max(100, max_decode_for_this_req // 2), max_decode_for_this_req)
            
            # --- Keep original workloads for baseline comparison ---
            elif workload_type == 'long':
                # Homogeneous batch of long prefill, long decode. Best-case for dot-product.
                prefill_len = rng.randint(max(1, max_prefill // 2), max_prefill)
                max_decode_for_this_req = max_target - prefill_len
                decode_len = rng.randint(max(1, max_decode_for_this_req // 2), max_decode_for_this_req)
            
            else: # 'realistic' (the original log-normal statistical average)
                prefill_len = int(rng.lognormvariate(mu=6.0, sigma=1.0))
                prefill_len = max(16, min(prefill_len, max_prefill))
                max_decode_for_this_req = max_target - prefill_len
                decode_len = int(rng.lognormvariate(mu=4.5, sigma=0.8))

            # --- Final validation for all workloads ---
            if max_decode_for_this_req <= 1: max_decode_for_this_req = 10
            decode_len = max(1, min(decode_len, max_decode_for_this_req))

            # Padding logic remains the same
            padded_len = 2 ** max(7, prefill_len.bit_length())
            padded_len = min(padded_len, max_prefill)
            tokens = jnp.zeros((padded_len,), dtype=jnp.int32)

            req = Request(
                id=i, prefill_len=prefill_len, decode_len=decode_len,
                arrival_time=0.0, tokens=tokens, true_length=prefill_len
            )
            requests.append(req)

        print(f"Generated {len(requests)} requests")
        print(f"  Avg prefill length: {np.mean([r.prefill_len for r in requests]):.1f}")
        print(f"  Avg decode length: {np.mean([r.decode_len for r in requests]):.1f}")
        return requests

    def _process_prefill_batch(self):
        """Process a batch of prefill requests"""
        while self.free_slots and not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
            except queue.Empty:
                break
            
            slot = self.free_slots.pop(0)
            request.slot_used = slot
            request.prefill_start_time = time.perf_counter()

            prefix, result_tokens = self.engine.prefill(
                params=self.params, padded_tokens=request.tokens,
                true_length=request.true_length, slot=slot
            )
            self.decode_state = self.engine.insert(
                prefix=prefix, decode_state=self.decode_state, slot=slot
            )
            
            request.prefill_end_time = time.perf_counter()
            request.first_token_time = request.prefill_end_time
            request.decode_start_time = request.prefill_end_time
            
            first_token = result_tokens.convert_to_numpy().data[0, 0].item()
            request.output_tokens.append(first_token)
            
            self.active_slots[slot] = request
            
            prefill_latency = request.prefill_end_time - request.prefill_start_time
            self.metrics.prefill_latencies.append(prefill_latency)
            self.metrics.time_to_first_token.append(request.first_token_time - request.arrival_time)
            self.metrics.total_prefill_tokens += request.prefill_len

    def _process_decode_step(self):
        """Process one decode step for all active requests"""
        if not self.active_slots:
            return
        
        decode_start = time.perf_counter()
        self.decode_state, result_tokens = self.engine.generate(
            params=self.params, decode_state=self.decode_state
        )
        decode_end = time.perf_counter()
        self.metrics.decode_step_latencies.append(decode_end - decode_start)
        self.last_decode_time = decode_end
        self.decode_step_count += 1
        
        result_np = result_tokens.convert_to_numpy()
        completed_slots = []
        
        for slot, request in list(self.active_slots.items()):
            if slot >= result_np.data.shape[0]: continue
            
            token = result_np.data[slot, 0].item()
            request.output_tokens.append(token)
            
            if len(request.output_tokens) - 1 >= request.decode_len:
                request.decode_end_time = decode_end
                request.completion_time = decode_end
                completed_slots.append(slot)
        
        for slot in completed_slots:
            request = self.active_slots.pop(slot)
            request.tokens = None # Release JAX array reference
            self.completed_requests.append(request)
            self.free_slots.append(slot)
            self.engine.free_resource(slot)
            
            decode_latency = request.decode_end_time - request.decode_start_time
            request_latency = request.completion_time - request.arrival_time
            self.metrics.decode_latencies.append(decode_latency)
            self.metrics.request_latencies.append(request_latency)
            self.metrics.total_decode_tokens += len(request.output_tokens) - 1
            self.metrics.completed_requests += 1
        
        utilization = len(self.active_slots) / self.engine.max_concurrent_decodes if self.engine.max_concurrent_decodes > 0 else 0
        self.metrics.slot_utilization_history.append((decode_end, utilization))
        
        current_active_tokens = sum(r.prefill_len + len(r.output_tokens) for r in self.active_slots.values())
        self.metrics.max_concurrent_tokens = max(self.metrics.max_concurrent_tokens, current_active_tokens)

        if self.decode_step_count % 100 == 0:
            print(f"Decode step {self.decode_step_count}: {len(self.active_slots)} active slots, "
                  f"{len(completed_slots)} completed this step")

    def _decode_loop(self):
        """Main decode loop running in separate thread"""
        while self.is_running:
            if not self.request_queue.empty(): self._process_prefill_batch()
            if self.active_slots: self._process_decode_step()
            elif self.all_requests_submitted and self.request_queue.empty(): break
            else: time.sleep(0.001)

    def _metrics_loop(self):
        """Metrics collection loop"""
        last_tokens, last_time = 0, time.perf_counter()
        while self.is_running:
            time.sleep(1.0)
            current_time = time.perf_counter()
            current_tokens = self.metrics.total_prefill_tokens + self.metrics.total_decode_tokens
            if current_tokens > last_tokens:
                throughput = (current_tokens - last_tokens) / (current_time - last_time)
                self.metrics.tokens_per_second_history.append((current_time, throughput))
                print(f"[Metrics] Throughput: {throughput:.1f} tok/s, Active: {len(self.active_slots)}, "
                      f"Completed: {self.metrics.completed_requests}/{self.metrics.total_requests}")
            last_tokens, last_time = current_tokens, current_time

    def run_benchmark(self, requests: List[Request]) -> BenchmarkMetrics:
        """Run the benchmark with given requests"""
        print(f"\nStarting benchmark with {len(requests)} requests...")
        self.metrics = BenchmarkMetrics()
        self.metrics.total_requests = len(requests)
        self.active_slots.clear()
        self.completed_requests.clear()
        self.free_slots = list(range(self.engine.max_concurrent_decodes))
        self.decode_step_count = 0
        
        self.metrics.benchmark_start_time = time.perf_counter()
        for req in requests:
            req.arrival_time = self.metrics.benchmark_start_time
            self.request_queue.put(req)

        self.is_running = True
        self.all_requests_submitted = True
        self.decode_thread = threading.Thread(target=self._decode_loop, name="decode_thread")
        self.metrics_thread = threading.Thread(target=self._metrics_loop, name="metrics_thread")
        self.decode_thread.start()
        self.metrics_thread.start()
        self.decode_thread.join()
        self.is_running = False
        self.metrics_thread.join()
        self.metrics.benchmark_end_time = time.perf_counter()
        print(f"Benchmark complete. Processed {self.metrics.completed_requests} requests")
        return self.metrics

def main(argv):
    parser = argparse.ArgumentParser(description="Controlled E2E Benchmark for MaxText",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('pyconfig_args', nargs='*', 
                        help="MaxText configuration arguments in key=value format.")
    parser.add_argument('--config_path', type=str, default="MaxText/configs/base.yml", 
                        help="Base config file path.")
    parser.add_argument('--num_requests', type=int, default=512, 
                        help="Number of requests to process.")
    parser.add_argument('--warmup_steps', type=int, default=10, 
                        help="Number of warmup decode steps.")
    parser.add_argument('--workload_type', 
                    choices=['uniform', 'short', 'long', 'realistic', 'chat', 'rag', 'summarize'], 
                    default='realistic', help="Type of workload distribution.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--output_file', type=str, default=None,
                        help="Output file for detailed results.")
    parser.add_argument('--skip_warmup', action='store_true', help="Skip warmup phase.")

    args = parser.parse_args(argv[1:])

    pyconfig_argv = [argv[0], args.config_path] + args.pyconfig_args
    print("Initializing MaxText configuration...")
    config = pyconfig.initialize(pyconfig_argv)
    
    benchmark_config = {
        'num_requests': args.num_requests,
        'warmup_steps': 0 if args.skip_warmup else args.warmup_steps,
        'workload_type': args.workload_type,
        'seed': args.seed,
    }
    
    print("\nCreating MaxEngine...")
    engine = maxengine.MaxEngine(config)
    
    benchmark = ControlledBenchmark(engine, config, benchmark_config)
    benchmark.load_and_warmup()
    requests = benchmark.generate_workload(
        num_requests=benchmark_config['num_requests'],
        seed=benchmark_config['seed']
    )
    metrics = benchmark.run_benchmark(requests)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    summary = metrics.get_summary()
    
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Attention: {config.attention}")
    print(f"  Batch Size (Global): {engine.max_concurrent_decodes}")
    print(f"  Max Prefill: {config.max_prefill_predict_length}")
    print(f"  Max Target: {config.max_target_length}")
    if config.attention == 'paged':
        print(f"  Paged Pages: {config.pagedattn_num_pages}")
        print(f"  Tokens/Page: {config.pagedattn_tokens_per_page}")
    
    print(f"\nWorkload:")
    print(f"  Type: {benchmark_config['workload_type']}")
    print(f"  Requests: {summary['total_requests']}")
    print(f"  Completed: {summary['completed_requests']}")
    
    print(f"\nPerformance:")
    print(f"  Total Time: {summary['total_time_s']:.2f} seconds")
    print(f"  Total Tokens: {summary['total_tokens']:,}")
    print(f"  Max Concurrent Active Tokens: {metrics.max_concurrent_tokens:,}")
    print(f"  Overall Throughput: {summary['throughput_tokens_per_s']:.2f} tokens/sec")
    print(f"  Prefill Throughput: {summary['prefill_throughput_tokens_per_s']:.2f} tokens/sec")
    print(f"  Decode Throughput: {summary['decode_throughput_tokens_per_s']:.2f} tokens/sec")
    
    print(f"\nLatencies:")
    print(f"  Prefill (avg): {summary['avg_prefill_latency_ms']:.1f} ms")
    print(f"  Prefill (p50): {summary['p50_prefill_latency_ms']:.1f} ms")
    print(f"  Prefill (p99): {summary['p99_prefill_latency_ms']:.1f} ms")
    print(f"  Time per Decode Step (avg): {summary['avg_time_per_decode_step_ms']:.2f} ms")
    print(f"  Time per Decode Step (p50): {summary['p50_time_per_decode_step_ms']:.2f} ms")
    print(f"  Time per Decode Step (p99): {summary['p99_time_per_decode_step_ms']:.2f} ms")
    print(f"  Inter-Token Decode Latency (avg): {summary['avg_decode_latency_s']:.2f} s")
    print(f"  Inter-Token Decode Latency (p50): {summary['p50_decode_latency_s']:.2f} s")
    print(f"  Inter-Token Decode Latency (p99): {summary['p99_decode_latency_s']:.2f} s")
    print(f"  Time to First Token (avg): {summary['avg_time_to_first_token_ms']:.1f} ms")
    print(f"  Time to First Token (p50): {summary['p50_time_to_first_token_ms']:.1f} ms")
    print(f"  Time to First Token (p99): {summary['p99_time_to_first_token_ms']:.1f} ms")
    print(f"  Request Latency (avg): {summary['avg_request_latency_s']:.2f} s")
    print(f"  Request Latency (p50): {summary['p50_request_latency_s']:.2f} s")
    print(f"  Request Latency (p99): {summary['p99_request_latency_s']:.2f} s")
    
    if args.output_file:
        # Create directory for output file if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        detailed_results = {
            'config': {
                'model': config.model_name,
                'attention': config.attention,
                'batch_size': engine.max_concurrent_decodes,
                'max_prefill': config.max_prefill_predict_length,
                'max_target': config.max_target_length,
                'pagedattn_num_pages': config.pagedattn_num_pages if config.attention == 'paged' else None,
                'pagedattn_tokens_per_page': config.pagedattn_tokens_per_page if config.attention == 'paged' else None,
            },
            'memory_usage': {
                'after_load_gb': metrics.mem_after_load_gb,
                'after_init_gb': metrics.mem_after_init_gb,
            },
            'benchmark_config': benchmark_config,
            'summary': summary,
            'peak_active_tokens': metrics.max_concurrent_tokens,
            'requests': [
                {
                    'id': r.id, 'prefill_len': r.prefill_len, 'decode_len': r.decode_len,
                    'output_len': len(r.output_tokens),
                    'prefill_latency_ms': (r.prefill_end_time - r.prefill_start_time) * 1000 
                        if r.prefill_end_time and r.prefill_start_time else None,
                    'decode_latency_s': (r.decode_end_time - r.decode_start_time) 
                        if r.decode_end_time and r.decode_start_time else None,
                    'total_latency_s': (r.completion_time - r.arrival_time) 
                        if r.completion_time else None,
                }
                for r in benchmark.completed_requests
            ],
            'throughput_history': metrics.tokens_per_second_history,
            'slot_utilization_history': metrics.slot_utilization_history,
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output_file}")
    
    print("="*60)

if __name__ == "__main__":
    import sys
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    if os.path.exists("/mnt/disks/persist/jax_cache"):
        jax.config.update("jax_compilation_cache_dir", "/mnt/disks/persist/jax_cache")
    main(sys.argv)