import datetime
import jax
import random
import string


def simple_timeit(f, *args, tries = 10, task = None):
    '''Simple utility to time a function for multiple runs'''
    assert task is not None

    trace_name = f"t_{task}_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    trace_dir = f"gs://runner-maxtext-logs/megablox/{trace_name}"

    outcomes_ms = []
    jax.block_until_ready(f(*args)) #warm it up!
    jax.profiler.start_trace(trace_dir)

    s = datetime.datetime.now()
    outputs = []
    for _ in range(tries):
        outputs.append(f(*args))

    jax.block_until_ready(outputs)
    e = datetime.datetime.now()
    jax.profiler.stop_trace()

    # for _ in range(tries):
    #     s = datetime.datetime.now()
    #     jax.block_until_ready(f(*args))
    #     e = datetime.datetime.now()
    #     outcomes_ms.append(1000*(e-s).total_seconds())
    # jax.profiler.stop_trace()

    # average_time_ms = sum(outcomes_ms)/len(outcomes_ms)
    average_time_ms = 1000 * (e-s).total_seconds() / tries
    print(f"{task}: average time milliseconds: {average_time_ms:.2f}, trace {trace_dir}")
    return average_time_ms