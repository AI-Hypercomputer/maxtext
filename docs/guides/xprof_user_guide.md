# XProf for MaxText developers


## Introduction to XProf

You can use XProf to profile and analyze the training performance of AI models. XProf helps you understand how to optimize model performance, identify bottlenecks, and improve training efficiency.


## Profiling in JAX 

XProf supports profiling JAX models, which is crucial for MaxText developers working with JAX. You can profile your JAX models using various methods, including:



*   **Sampling Mode** This mode allows for continuous profiling by sampling data during model execution.
*   **Programmatic Mode:** This provides more granular control over when and what to profile, allowing you to instrument your code with specific profiling markers. This method is integrated with MaxText code.


Below is an example of how programmatic XProf works with Python code.


```
JAX.profiler.trace(profile_diretory)
batch = next(data_iterator)
with JAX.profiler.StepTraceAnnotation("train_step", step_num=i):
  state, metrics = train_step(state, batch)
JAX.profiler.stop()
```


`JAX.profiler.TraceAnnotation` is a tool for adding custom annotations to JAX traces.


## For more detailed tool illustration see 
[XProf Detained Feature Document from OpenXLA](https://openxla.org/XProf)



