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
# change 2
import jax
import jax.numpy as jnp
from flax import struct

# I hope these are universally good, so I'm hardcoding.
cfg_max_count = 5.0 # stiffness UCB
cfg_std_count = 6.0 # This is equivalent to 5 standard deviations, the level at which we announce findings as discoveries in physics.


class Stats(struct.PyTreeNode):
  count: jnp.ndarray
  sum1: jnp.ndarray
  dev: jnp.ndarray

  @classmethod
  def make(cls, p):
    count = jnp.array(0.0, dtype=jnp.float32)
    sum1 = jnp.array(0.0, dtype=jnp.float32)
    dev = jnp.array(0.0, dtype=jnp.float32)
    return cls(count=count, sum1 = sum1, dev = dev)

  def update(self, sample):
    count_p_1 = self.count + 1.0
    decay = jnp.minimum(1.0, cfg_max_count / count_p_1)
    new_count = count_p_1 * decay
    new_sum1 = jnp.add(self.sum1, sample) * decay
    new_mean = new_sum1 / new_count
    new_dev = jnp.add(self.dev, jnp.abs(sample - new_mean)) * decay
    return Stats(count=new_count, sum1=new_sum1, dev=new_dev)

  def stats(self):
    count = jnp.maximum(self.count, 1.0)
    mean = self.sum1 / count
    dev = self.dev / count
    return mean, dev

  def rescale(self, sample):
    mean, dev = self.stats()
    ucb = mean + cfg_std_count * dev
    rescale = jnp.minimum(ucb, sample) / sample
    # Do not rescale for some time
    rescale = jax.lax.select(self.count < cfg_max_count / 2.0, jnp.ones_like(rescale), rescale)
    return rescale

  def metrics(self):
    mean, dev = self.stats()
    return {
      'ucb/mean': mean,
      # 'ucb/ucb': ucb,
      'ucb/dev' : dev,
      # 'ucb/rescale': rescale,
    }

  def str(self):
    return f'count={self.count:3.1f}, sum1={self.sum1:3.1f}, dev={self.dev:3.1f}'


def ucb_init(params):
  return jax.tree_map(lambda p: Stats.make(p), params)


def ucb_update(stats, grads):
  # sample = jnp.linalg.norm(grads, ord=2)
  # rescale = stats.rescale(sample)
  # new_grads = jax.tree_map(lambda g, r: g*r, grads, rescale)
  # new_stats = stats.update(sample * rescale)
  # metrics = new_stats.metrics()
  def metrics(r, nst):
    metrics = nst.metrics()
    metrics['rescale'] = r
    return metrics

  sample    = jax.tree_map(lambda g: jnp.sqrt(jnp.sum(g*g)), grads)
  rescale   = jax.tree_map(lambda sa, st: st.rescale(sa), sample, stats)
  new_grads = jax.tree_map(lambda g, r: g*r.astype(g.dtype), grads, rescale)
  new_stats = jax.tree_map(lambda sa, r, st: st.update(sa*r), sample, rescale, stats)
  metrics   = jax.tree_map(metrics, rescale, new_stats)
  return new_stats, new_grads, metrics


def test():
  samples = []

  def val(v):
    return jnp.full([1], v, dtype=jnp.bfloat16), jnp.full([4], 2*v, dtype=jnp.bfloat16)

  stats = ucb_init(val(0))

  def update(grads):
    nonlocal stats
    # mean, dev = stats.stats()
    # mean, dev = float(mean), float(dev)
    # ucb = float(mean + cfg_std_count * dev)
    # exp_mean = float(jnp.mean(jnp.array(samples)))
    # exp_dev  = float(jnp.mean(jnp.abs(jnp.array(samples) - exp_mean)))
    # exp_ucb = float(exp_mean + cfg_std_count * exp_dev)
    # print(
    #   f'sample={grads.astype(jnp.float32)[0]: 5.1f} '
    #   f'ucb ({exp_ucb: 11.8f} - {ucb: 11.8f} = {exp_ucb - ucb: 11.8f}); '
    #   f'mean ({exp_mean: 11.8f} - {mean: 11.8f}) '
    #   f'mean ({exp_dev: 11.8f} - {dev: 11.8f}) '
    # )
    # samples.append(jnp.linalg.norm(grads, ord=2))
    # return new_grads

  print(stats)
  print()
  def update(v):
    nonlocal stats
    grads = val(v)
    stats, new_grads, metrics = ucb_update(stats, grads)
    # print(stats)
    print(metrics)
    print("grads      = ", grads)
    print("clip_grads = ", new_grads)
    print()

  for i in range(int(cfg_max_count*2)):
    update(i+1)
  update(100.0)
    # assert (grads == new_grads).all()

  # new_grads = update(val(100))
  # assert (new_grads == val(0.0)).all()
  # new_grads = update(val(100))
  # assert (new_grads == val(0.0)).all()
  # new_grads = update(val(100))
  # assert (new_grads == val(0.0)).all()

  # new_grads = update(val(100))
  # assert (new_grads == val(100)).all()

# test()
