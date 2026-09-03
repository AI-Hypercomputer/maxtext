"""Sort activations tests."""

import contextlib
import functools

from absl import flags
import jax
import jax.numpy as jnp
from maxtext.kernels import sort_activations

import google3.perftools.accelerators.xprof.api.python as xprof
from google3.testing.pybase import googletest
from google3.testing.pybase import parameterized


_ENABLE_XPROF_TRACING = flags.DEFINE_bool(
    "enable_xprof_tracing",
    default=False,
    help="If true, generate xprof traces of the tests.",
)


@contextlib.contextmanager
def _maybe_xprof_session(**kwargs):
  """A context manager that enables xprof tracing only if the flag is set."""
  if _ENABLE_XPROF_TRACING.value:
    with xprof.session(**kwargs) as session:
      yield session
  else:
    yield None


class SortActivationsTest(parameterized.TestCase):

  def test_route_output(self):
    self.assertTrue(
        jnp.array_equal(
            sort_activations.route(
                tokens=jnp.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                    ]
                ),
                selected_experts=jnp.array(
                    [
                        [0, 5],
                        [2, 3],
                    ]
                ),
                use_gather_mosaic_kernel=False,
            ),
            jnp.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [4, 5, 6],
                    [1, 2, 3],
                ]
            ),
        )
    )

  def test_unroute_output(self):
    self.assertTrue(
        jnp.array_equal(
            sort_activations.unroute(
                tokens=jnp.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12],
                    ]
                ),
                selected_experts=jnp.array(
                    [
                        [0, 5],
                        [2, 3],
                    ]
                ),
                use_gather_mosaic_kernel=False,
            ),
            jnp.array(
                [
                    [1 + 10, 2 + 11, 3 + 12],
                    [4 + 7, 5 + 8, 6 + 9],
                ]
            ),
        )
    )

  @parameterized.named_parameters(
      dict(  # pylint: disable=use-dict-literal
          testcase_name="tiny",
          num_tokens=3,
          model_dim=7,
          num_selections=5,
          num_experts=256,
          use_gather_mosaic_kernel=False,
      ),
      dict(  # pylint: disable=use-dict-literal
          testcase_name="reference",
          num_tokens=16 * 1024,
          model_dim=7 * 1024,
          num_selections=8,
          num_experts=256,
          use_gather_mosaic_kernel=False,
      ),
  )
  def test_route_unroute_vjp(
      self,
      num_tokens,
      model_dim,
      num_selections,
      num_experts,
      use_gather_mosaic_kernel,
  ):
    key = jax.random.PRNGKey(0)
    tokens = jax.random.normal(key, shape=(num_tokens, model_dim))
    selected_experts = jax.random.permutation(
        key,
        jnp.broadcast_to(jnp.arange(num_experts), (num_tokens, num_experts)),
        axis=-1,
        independent=True,
    )[:, :num_selections]

    @functools.partial(jax.jit, static_argnames=("use_gather_mosaic_kernel",))
    def loss_fn(x, selected_experts, use_gather_mosaic_kernel):
      x = sort_activations.route(x, selected_experts, use_gather_mosaic_kernel)
      x *= jnp.repeat(
          jnp.arange(num_experts),
          repeats=jnp.bincount(jnp.ravel(selected_experts), length=num_experts),
          total_repeat_length=selected_experts.shape[0] * selected_experts.shape[1],
      )[:, None]
      x = sort_activations.unroute(x, selected_experts, use_gather_mosaic_kernel)
      return jnp.sum(x), x

    with _maybe_xprof_session(trace_python=True) as xprof_url:
      grad, output = jax.grad(loss_fn, has_aux=True)(tokens, selected_experts, use_gather_mosaic_kernel)

    if xprof_url:
      print(f"Xprof URL: {xprof_url}")

    with self.subTest("output"):
      expected = tokens * jnp.sum(selected_experts, axis=-1, keepdims=True)
      self.assertTrue(
          jnp.allclose(output, expected),
          msg=(
              f"output =\n{output}\n\n"  #
              f"expected =\n{expected}\n\n"  #
              f"tokens =\n{tokens}\n\n"  #
              f"selected_experts =\n{selected_experts}\n\n"
          ),
      )

    with self.subTest("gradient"):
      expected = jnp.broadcast_to(jnp.sum(selected_experts, axis=-1, keepdims=True), tokens.shape)
      self.assertTrue(
          jnp.array_equal(grad, expected),
          msg=(
              f"grad =\n{grad}\n\n"  #
              f"expected =\n{expected}\n\n"  #
              f"tokens =\n{tokens}\n\n"  #
              f"selected_experts =\n{selected_experts}\n\n"
          ),
      )


if __name__ == "__main__":
  googletest.main()
