"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import jax
import jax.numpy as jnp
from flax import struct
import max_utils

class Stats(struct.PyTreeNode):
  count: jnp.ndarray
  sum1: ...
  sum2: ...
  max_count: int | None

  @classmethod
  def make(cls, tree, max_count, init_count = 1.0, init_sum1 = 0.0, init_sum2 = 1.0):
    count = jnp.array(init_count, dtype=jnp.float32)
    sum1 = jax.tree_map(lambda x: jnp.ones(x.shape, dtype=jnp.bfloat16) * init_sum1, tree)
    sum2 = jax.tree_map(lambda x: jnp.ones(x.shape, dtype=jnp.bfloat16) * init_sum2, tree)
    return cls(
        count=count,
        max_count=max_count,
        sum1 = sum1,
        sum2 = sum2,
    )

  def update(self, sample):
    count_p_1 = self.count + 1.0
    decay = jnp.minimum(1.0, self.max_count / count_p_1)

    def reduce(x):
      return jnp.mean(x, keepdims=True)

    new_count = count_p_1 * decay
    # print('nc', new_count, count_p_1,  decay)
    new_sum1 = jax.tree_map(lambda acc, s: jnp.add(acc, reduce(s   )) * decay, self.sum1, sample)
    new_sum2 = jax.tree_map(lambda acc, s: jnp.add(acc, reduce(s**2)) * decay, self.sum2, sample)
    return Stats(count=new_count, max_count=self.max_count, sum1=new_sum1, sum2=new_sum2)

  def upper_confidence_bound(self):
    def ucb(s, s2):
      mean = s / self.count
      mean2 = s2 / self.count
      std = jnp.sqrt(mean2 - mean**2)
      return mean, std
      # g > mean + std_count * std ?
      # g - mean / std > 4 ? - filter or clip these
      # adam: mean / sqrt(mean2)

    return jax.tree_map(ucb, self.sum1, self.sum2)

# I hope these are universally good, so I'm hardcoding.
cfg_max_count = 20.0
cfg_std_count = 4.0

def ucb_init(max_count=cfg_max_count):
  return Stats.make(jnp.zeros([]), max_count, init_count=0.001, init_sum1=0.0, init_sum2=0.0)


def ucb_update(stats, grads):
  grads_l2 = max_utils.l2norm_pytree(grads)
  grad_mean, grad_std = stats.upper_confidence_bound()
  grad_ucb = grad_mean + cfg_std_count * grad_std
  is_spike = grads_l2 > grad_ucb
  warmup_done = stats.count >= stats.max_count - 1.0
  zero_out = is_spike & warmup_done
  # jax.debug.print("   {} {} {}  Q", is_spike, warmup_done, zero_out)
  new_grads = jax.lax.cond(zero_out, lambda: jax.tree_map(jnp.zeros_like, grads), lambda: grads)

  new_stats = stats.update(grads_l2)
  return new_stats, new_grads, (grad_mean, grad_std, grad_ucb, zero_out)


def test():
  # tree = {'a': jnp.array(0.0), 'b': jnp.array([1.0, 1.0])}
  tree = {'a': jnp.array(0.0) }
  # tree = jnp.array(0.0)
  stats = Stats.make(tree, init_count=0, init_sum1=0.0, init_sum2=0.0, max_count=30)
  for i in range(100):
    # stats = stats.update({'a': jnp.array(float(i+1)), 'b': jnp.array([(float(i+2)), (float(i+2))])})
    stats = stats.update({'a': jnp.array(float(i+1))})
    # stats = stats.update(jnp.array(float(i+1)))
    std_count = 2.0
    mean, std = stats.upper_confidence_bound()
    ucb = mean + std_count * std
    elts = jnp.array(list(range(i+1))) + 1
    ucb_a = ucb['a']
    # ucb_a = ucb
    print(f'i+1={i+1}; ucb should be: {jnp.mean(elts) + std_count*jnp.std(elts):2.8f} but is {ucb_a[1]:2.8f}; mean {ucb_a[0]: 2.8f}')
    # print(stats, '\n')

# print('test', flush=True)
# test()