# Xprof User Guide for Maxtext Developers


## Introduction to Xprof

Xprof is a powerful tool designed for profiling and analyzing the training performance of AI models. For Maxtext developers, understanding and utilizing Xprof can significantly help in optimizing model performance, identifying bottlenecks, and improving training efficiency.


## Profiling Jax 

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


## Xprof UI Features

The Xprof User Interface (UI) offers several key features to help you analyze your profiling data:



*   Trace Viewer

     Visualize the execution timeline of your model, allowing you to see the duration of different operations and identify sequential dependencies.


    At the XLA Operations level, you can click on each HLO operation to view its shape, wall-duration, and timing statistics.


*   Graph Viewer

    A High-Level Optimizer (HLO) graph browser tool. It enables a graph view to visualize HLOs with their input and output compute flow.



*   HLO Profile

     
    Provides insights into the HLO operations, including their costs and execution times.


    Top overall FLOPS/bandwidth utilization showed weight average across all ops in the captured graph.


    You can click each individual ops, which shows individual op flops utilization and HBM utilization.

    	

*   HLO Op Stats

    roofline analysis of operations

*   Roofline Model

    You can view both Program-Level Analysis and operation-level analysis


    `achievable FLOP/s = min(peak FLOP/s, peak BW \* operational intensity)`


    `operational intensity = FLOPs / bytes`


    For each HLO operation, you can view FLOPs rate, HBM rate, and roofline efficiency. It also indicates whether the operation is HBM or compute-bound based on these calculations.



*   **Memory Viewer**

    Analyze memory usage patterns during model training, crucial for optimizing memory-intensive operations.

    From this tab, you can view each buffer allocation's source code, shape, and spanned time frame.



