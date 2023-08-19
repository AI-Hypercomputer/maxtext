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
  sum1: jnp.ndarray
  sum2: jnp.ndarray
  max_count: float | None

  @classmethod
  def make(cls, max_count: float, init_count: float = 1.0, init_sum1: float = 0.0, init_sum2: float = 1.0):
    count = jnp.array(init_count, dtype=jnp.float32)
    sum1 = jnp.array(init_sum1, dtype=jnp.float32)
    sum2 = jnp.array(init_sum2, dtype=jnp.float32)
    return cls(
        count=count,
        sum1 = sum1,
        sum2 = sum2,
        max_count=max_count,
    )

  def update(self, sample):
    count_p_1 = self.count + 1.0
    decay = jnp.minimum(1.0, self.max_count / count_p_1)
    new_count = count_p_1 * decay
    new_sum1 = jnp.add(self.sum1, sample   ) * decay
    new_sum2 = jnp.add(self.sum2, sample**2) * decay
    return Stats(count=new_count, max_count=self.max_count, sum1=new_sum1, sum2=new_sum2)

  def stats(self):
    mean = self.sum1 / self.count
    mean2 = self.sum2 / self.count
    std = jnp.sqrt(mean2 - mean**2)
    return mean, std

  def update_and_detect_spike(self, sample, std_count):
    mean, std = self.stats()
    ucb = mean + std_count * std
    is_spike = sample > ucb
    warmup_done = self.count >= self.max_count - 1.0
    warm_spike = (is_spike & warmup_done)
    new_stats = self.update(sample)
    metrics = {
      'ucb/mean': mean,
      'ucb/ucb': ucb,
      'ucb/std' : std,
      'ucb/is_spike': is_spike,
      'ucb/warm_spike': warm_spike,
    }
    return new_stats, warm_spike, metrics


# I hope these are universally good, so I'm hardcoding.
cfg_init_count = 0.000
cfg_max_count = 20.0
cfg_std_count = 4.0


def ucb_init(max_count=cfg_max_count, init_count=cfg_init_count):
  return Stats.make(max_count, init_count=init_count, init_sum1=0.0, init_sum2=0.0)


def ucb_update(stats, grads):
  grads_l2 = max_utils.l2norm_pytree(grads)
  new_stats, zero_out, metrics = stats.update_and_detect_spike(grads_l2, cfg_std_count)
  def update_tensor(t):
    mask = jnp.full(t.shape, zero_out, dtype = jnp.bool_)
    return jax.lax.select(mask, jnp.zeros_like(t), t)
  new_grads = jax.tree_map(update_tensor, grads)
  new_stats = stats.update(grads_l2)
  # print('QQ' , type(grads) , type(new_grads))
  return new_stats, new_grads, metrics


def test():
  samples = []
  stats = ucb_init(init_count = 0.0)
  def update(grads):
    nonlocal stats
    mean, std = stats.stats()
    mean, std = float(mean), float(std)
    stats, new_grads, metrics = ucb_update(stats, grads)
    ucb = float(mean + cfg_std_count * std)
    exp_mean = float(jnp.mean(jnp.array(samples)))
    exp_ucb = float(exp_mean + cfg_std_count * jnp.std(jnp.array(samples)))
    print(
      f'sample={grads.astype(jnp.float32)[0]: 5.1f} '
      f'ucb ({exp_ucb: 11.8f} - {ucb: 11.8f} = {exp_ucb - ucb: 11.8f}); mean ({exp_mean: 11.8f} - {mean: 11.8f}) '
    )
    samples.append(max_utils.l2norm_pytree(grads))
    return new_grads

  def f(v):
    return jnp.full([1], v, dtype=jnp.bfloat16)
  for i in range(int(cfg_max_count + 5.0)):
    # grads = jnp.full([2], float(i+1))
    grads = f(i+1)
    new_grads = update(grads)
    assert (grads == new_grads).all()
  new_grads = update(f(45))
  assert (new_grads == f(0.0)).all()

  new_grads = update(f(45))
  assert (new_grads == f(45)).all()

# test()