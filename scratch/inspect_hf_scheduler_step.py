import inspect
from diffusers import DDIMScheduler

scheduler = DDIMScheduler()
print(inspect.getsource(scheduler.step))
