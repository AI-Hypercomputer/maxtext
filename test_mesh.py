import jax
from jax.sharding import Mesh
import numpy as np
devices = np.array(jax.devices()).reshape((1, 4))
mesh = Mesh(devices, ('data', 'model'))
print("Mesh1 axis_names:", mesh.axis_names)
devices2 = devices.reshape((1, 1, 4, 1, 1))
mesh2 = Mesh(devices2, ('data', 'attn_dp', 'model', 'expert', 'attn_dp_expert'))
print("Mesh2 axis_names:", mesh2.axis_names)
