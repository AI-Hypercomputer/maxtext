import asyncio.selector_events
import ray
import traceback
import os
import jax
import random
import redis
import datetime
import asyncio
from contextlib import contextmanager
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy as NASS

import max_logging

class RayClusterCoordinator:
    def __init__(self, worker_cls, hang_time_threshold) -> None:
        self.worker_cls = worker_cls
        self.num_workers = int(os.environ.get('NGPUS'))
        self.num_workers_per_node = int(os.environ.get('GPUS_PER_NODE'))
        self.workers_initialized = False
        self.log = lambda user_str: max_logging.log(f"[RayClusterCoordinator] {user_str}")
        self.hang_time_threshold = hang_time_threshold if hang_time_threshold is not None else 300

        self.redis_addr = os.environ.get('REDIS_ADDR').split(':')

        worker_node_info, self.num_physical_nodes = self._get_schedulable_worker_info()
        self.workers = [worker_cls.options(num_gpus=1,
                                           num_cpus=16,
                                           resources={"worker_units": 1},
                                           scheduling_strategy=NASS(node_id=worker_node_info[i][0], soft=False)).remote(i, 
                                                                                                                        worker_node_info[i][1],
                                                                                                                        worker_node_info[i][2])
                                           for i in range(self.num_workers)]

        self.jax_coordinator_ip = worker_node_info[0][2]
        self.redis = redis.Redis(host=self.redis_addr[0], port=int(self.redis_addr[1]), decode_responses=True, password=None)
        self._init_sync_dict()
    
    def _get_schedulable_worker_info(self):
        worker_node_info = []
        worker_nodes = sorted([node for node in ray.nodes() if (node['Alive'] and 'worker_units' in node['Resources'])], 
                              key=lambda x: x['NodeID'])

        num_nodes_required = self.num_workers // self.num_workers_per_node
        num_nodes_available = len(worker_nodes)
        assert num_nodes_required <= num_nodes_available
        
        worker_nodes = worker_nodes[:num_nodes_required]
        for worker_node_id, worker_node in enumerate(worker_nodes):
            for _ in range(self.num_workers_per_node):
                worker_node_info.append((worker_node['NodeID'], worker_node_id, worker_node['NodeName']))

        return worker_node_info, num_nodes_required

    def _init_sync_dict(self):
        self.redis.flushdb()
        init_time = datetime.datetime.now().isoformat()
        for pid in range(self.num_workers):
            self.redis.set(pid, init_time)

    def initialize_workers(self, **kwargs):
        self.worker_init_kwargs = kwargs
        coordinator_port = random.randint(1, 100000)  % 2**12 + (65535 - 2**12 + 1)
        self.jax_coordinator_addr = f"{self.jax_coordinator_ip}:{coordinator_port}"
        
        ray.get([w.initialize.remote(self.jax_coordinator_addr, self.num_workers, **kwargs) for i, w in enumerate(self.workers)])
        self.workers_initialized = True
    
    async def _run_workers_async(self, *args, **kwargs):
        worker_run_futures = [w.run.remote(*args, **kwargs) for w in self.workers]
        while True:
            completed_worker_results = []
            for _, wf in enumerate(worker_run_futures):
                try:
                    worker_result = ray.get(wf, timeout=0)
                    completed_worker_results.append(worker_result)
                except ray.exceptions.GetTimeoutError:
                    continue
            
            if len(completed_worker_results) < len(self.workers):
                self.log(f"All workers seem to be alive, but only {len(completed_worker_results)} completed")
                await asyncio.sleep(30)
            else:
                self.log(f"All {len(completed_worker_results)} workers completed. Returning results.")
                return completed_worker_results
    
    async def _detect_worker_hang_async(self):
        # Check if processes are hanging
        while True:
            await asyncio.sleep(30)
            for pid in range(self.num_workers):
                current_time = datetime.datetime.now()
                last_hearbeat_time = datetime.datetime.fromisoformat(self.redis.get(pid))
                elapsed = (current_time - last_hearbeat_time).total_seconds()
                if elapsed > self.hang_time_threshold:
                    self.log(f"Worker {pid} has been hanged for {elapsed / 60} minutes")
                    raise Exception(f"Worker {pid} appears to have hanged")

            self.log("No hangs detected")

    async def run(self, *args, **kwargs):
        if not self.workers_initialized:
            raise ValueError("""Cannot run workers without initializing them first. 
                                Please call the initialize_workers method of your cluster coordinator first.""")

        runners = asyncio.create_task(self._run_workers_async(*args, **kwargs))
        hang_detector = asyncio.create_task(self._detect_worker_hang_async())
        while True:
            try:
                done, _ = await asyncio.wait({runners, hang_detector}, return_when=asyncio.FIRST_COMPLETED)
                for task in done:    
                    # If the runner finish with exception first this will raise an exception
                    # If the hang detector finishes with exception first this will raise an exception
                    # The only case in which task.result() does not raise an exception is when
                    # the runners finish first without raising an exception. In that case
                    # get the results from the runners and cancel the hang detector task 
                    # before returning
                    result = task.result()
                    hang_detector.cancel()
                    return result
            except Exception as e:
                self.log(f"Encountered exception {type(e).__name__}")
                self.log(traceback.format_exc())
                
                self.log("Cancelling all tasks in event loop...")
                runners.cancel()
                hang_detector.cancel()
                self.log("Done cancelling all tasks in event loop")

                self.log("Killing all ray actors...")
                for w in self.workers:
                    ray.kill(w)
                self.workers_initialized = False
                del self.workers
                self.log("Done killing all ray actors")

                # Restart workers and reinitialize tasks
                self.log("Restarting all actors")
                worker_node_info, self.num_physical_nodes = self._get_schedulable_worker_info()
                self.workers = [self.worker_cls.options(num_gpus=1, 
                                                        num_cpus=16,
                                                        resources={"worker_units": 1},
                                                        scheduling_strategy=NASS(node_id=worker_node_info[i][0], soft=False)).remote(i, 
                                                                                                                                     worker_node_info[i][1],
                                                                                                                                     worker_node_info[i][2])
                                           for i in range(self.num_workers)]
                self.jax_coordinator_ip = worker_node_info[0][2]
                self._init_sync_dict()
                self.initialize_workers(**self.worker_init_kwargs)

                self.log("Reinitializing tasks")
                runners = asyncio.create_task(self._run_workers_async(*args, **kwargs))
                hang_detector = asyncio.create_task(self._detect_worker_hang_async())

class ResilientWorker:
    def __init__(self, process_id, physical_node_id, physical_node_ip):
        self.process_id = process_id
        self.physical_node_id = physical_node_id
        self.host_ip = physical_node_ip

        self.redis_addr = os.environ.get('REDIS_ADDR').split(':')
        self.logical_gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES'))
        self.redis = redis.Redis(host=self.redis_addr[0], port=int(self.redis_addr[1]), decode_responses=True, password=None)
    
    def get_process_id(self):
        return self.process_id
    
    def get_host_ip(self):
        return self.host_ip
     
    def get_logical_gpu_id(self):
        return self.logical_gpu_id

    def get_physical_node_id(self):
        return self.physical_node_id
    
    def initialize(self, coordinator_addr, num_processes):
        jax.distributed.initialize(coordinator_address=coordinator_addr, num_processes=num_processes, process_id=self.process_id, local_device_ids=0)

    def heartbeat(self):
        current_time = datetime.datetime.now().isoformat()
        self.redis.set(self.process_id, current_time)
    
    def run(self, *args, **kwargs):
        raise NotImplementedError