# Profiling with XProf



## Introduction to XProf

[XProf](https://openxla.org/xprof) is a profiling and performance analysis tool for machine learning.

You can use XProf to profile and analyze the training performance of AI models. XProf helps you understand how to optimize model performance, identify bottlenecks, and improve training efficiency.


## Profiling in JAX 

XProf supports profiling JAX models, which is crucial for MaxText developers working with JAX. You can profile your JAX models using various methods, including:

 
*   **Programmatic Mode:** This provides more granular control over when and what to profile, allowing you to instrument your code with specific profiling markers. This method is integrated with MaxText code.


The following example shows how to trace a JAX operation in Python.


```
import jax
import jax.numpy as jnp
import jax.profiler

profile_directory='dir_to_upload_profile'
jax.profiler.trace(profile_directory)
x = jnp.ones((1000, 1000))
iter = 0
with jax.profiler.StepTraceAnnotation("dot_product", step_num=iter):
    result = jnp.dot(x, x.T).block_until_ready()
    iter += 1
jax.profiler.stop()
```


You can use [`jax.profiler.TraceAnnotation`](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.TraceAnnotation.html) to add custom annotations to JAX traces.

*  **Sampling Mode** This mode allows for continuous profiling by sampling data during model execution.
This mode has not yet been enabled in MaxText yet. Refer to [remote-profiling](https://docs.jax.dev/en/latest/profiling.html#remote-profiling) for manual capture/sampling.

## Profiling configuration in MaxText

The following parameters control how profiling is executed within MaxText, allowing you to capture detailed performance data for analysis.

* `profiler`	specifies the profiler backend to use for capturing performance traces.	Options can be `xplane`, `nsys`. Default is "".	`xplane` is for XLA/TPU and `nsys` is for CUDA/GPU.

* `profiler_steps` defines the total number of steps to run during the profiling capture window.	Default is 5

* `skip_first_n_steps_for_profiler` specifies the number of initial training steps to skip before the profiling capture begins. This is typically used to bypass model warmup and capture steady-state performance.	default is 1.

For more information about XProf tools, see the [XProf documentation](https://openxla.org/xprof).




