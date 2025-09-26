# Xprof User Guide for Maxtext Developers


## Introduction to Xprof

Xprof is a powerful tool designed for profiling and analyzing the training performance of AI models. For Maxtext developers, understanding and utilizing Xprof can significantly help in optimizing model performance, identifying bottlenecks, and improving training efficiency.


## Profiling in Jax 

Xprof supports profiling Jax models, which is crucial for Maxtext developers working with Jax. You can profile your Jax models using various methods, including:



*   **Sampling Mode:** This mode allows for continuous profiling by sampling data during model execution.
*   **Programmatic Mode:** This provides more granular control over when and what to profile, allowing you to instrument your code with specific profiling markers. This method is integrated with Maxtext code.


Below is an example of how programmatic Xprof works with Python code.


```
jax.profiler.trace(profile_diretory)
batch = next(data_iterator)
with jax.profiler.StepTraceAnnotation("train_step", step_num=i):
  state, metrics = train_step(state, batch)
jax.profiler.stop()
```


`jax.profiler.TraceAnnotation` is a tool for adding custom annotations to JAX traces.


## For more detailed tool illustration see 
[Xprof Detained Feature Document from OpenXLA](https://openxla.org/xprof)



