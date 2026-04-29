import dataclasses
import functools
import numpy as np

import jax
import jax.numpy as jnp
import qwix
from qwix._src import interception
from qwix._src.core import dot_general as qwix_dot_general
from qwix._src.core import dot_general_qt as qwix_dot_general_qt
from qwix._src.core import numerics
from qwix._src.core import qarray
from qwix._src.providers.qt import QtProvider


def _sharding_of(x):
    if x is None:
        return None
    if hasattr(x, "sharding"):
        return x.sharding
    try:
        return jax.typeof(x).sharding
    except Exception:
        return None


def _spec_of(x):
    if x is None:
        return None
    sharding = x.spec if hasattr(x, "spec") else _sharding_of(x)
    if sharding is None:
        return None
    return sharding.spec if hasattr(sharding, "spec") else sharding


def _inferred_out_sharding(lhs, rhs, dimension_numbers):
    lhs_spec = _spec_of(lhs)
    rhs_spec = _spec_of(rhs)
    if lhs_spec is None or rhs_spec is None:
        return None

    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    lhs_ra = tuple(i for i in range(lhs.ndim) if i not in lhs_ca and i not in lhs_ba)
    rhs_ra = tuple(i for i in range(rhs.ndim) if i not in rhs_ca and i not in rhs_ba)

    out = []
    for l_axis, r_axis in zip(lhs_ba, rhs_ba):
        l_spec = lhs_spec[l_axis]
        r_spec = rhs_spec[r_axis]
        if l_spec is None:
            out.append(r_spec)
        elif r_spec is None or l_spec == r_spec:
            out.append(l_spec)
        else:
            raise ValueError(
                "Ambiguous batch-axis sharding in quantized dot_general: "
                f"{l_spec=} {r_spec=} {dimension_numbers=}"
            )
    out.extend(lhs_spec[i] for i in lhs_ra)
    out.extend(rhs_spec[i] for i in rhs_ra)
    return jax.P(*out)


def _pretranspose_out_sharding(target, transpose_axes):
    spec = _spec_of(target)
    if spec is None:
        return None
    inverse = tuple(np.argsort(transpose_axes))
    return jax.P(*(spec[i] for i in inverse))


@interception.disable_interceptions
def _dot_general_qt_fwd(
    lhs,
    rhs,
    lhs_calibration,
    rhs_calibration,
    dimension_numbers,
    config,
    out_sharding,
):
    lhs_in, rhs_in = lhs, rhs
    if lhs_calibration is not None:
        scale, zero_point = qarray.compute_scale_zero_point(lhs_calibration, config.lhs_qtype)
        lhs = qarray.quantize_with_scale_zero_point(lhs, config.lhs_qtype, scale, zero_point)
    if rhs_calibration is not None:
        scale, zero_point = qarray.compute_scale_zero_point(rhs_calibration, config.rhs_qtype)
        rhs = qarray.quantize_with_scale_zero_point(rhs, config.rhs_qtype, scale, zero_point)
    residuals = (lhs_in, rhs_in, lhs, rhs, lhs_calibration, rhs_calibration)
    return qwix_dot_general._slow_dot_general(lhs, rhs, dimension_numbers, out_sharding=out_sharding), residuals  # noqa: SLF001


def _dot_general_qt_bwd(dimension_numbers, config, out_sharding, residuals, g):
    del out_sharding
    lhs_in, rhs_in, lhs, rhs, lhs_calibration, rhs_calibration = residuals

    def _compute_gradient_for_operand(g, y, *, for_dlhs, target):
        bwd_dnums, transpose_axes = qwix_dot_general_qt._update_dimension_numbers_for_backward(  # noqa: SLF001
            dimension_numbers,
            (lhs.ndim, rhs.ndim),
            for_dlhs=for_dlhs,
        )
        if for_dlhs:
            g_qtype = config.dlhs_grad_qtype
            g_tile_size = config.dlhs_tile_size
            g_calibration_method = config.dlhs_grad_calibration_method
            g_noise_fn = config.dlhs_stochastic_rounding_noise_fn
        else:
            g_qtype = config.drhs_grad_qtype
            g_tile_size = config.drhs_tile_size
            g_calibration_method = config.drhs_grad_calibration_method
            g_noise_fn = config.drhs_stochastic_rounding_noise_fn

        if g_qtype and numerics.should_quantize(g.dtype):
            g_how = qwix_dot_general.get_how_to_quantize(
                dimension_numbers=bwd_dnums,
                ndims=(g.ndim, y.ndim),
                for_lhs=True,
                qtype=g_qtype,
                tile_size=g_tile_size,
                calibration_method=g_calibration_method,
                noise_fn=g_noise_fn,
            )
            disable_channelwise = config.lhs_disable_channelwise_axes if for_dlhs else config.rhs_disable_channelwise_axes
            if disable_channelwise:
                g_how = dataclasses.replace(g_how, channelwise_axes=[])
            g = qarray.quantize(g, g_how)

        grad_out_sharding = _pretranspose_out_sharding(target, transpose_axes)
        grad_res = qwix_dot_general._slow_dot_general(g, y, bwd_dnums, out_sharding=grad_out_sharding)  # noqa: SLF001
        return jax.lax.transpose(grad_res, transpose_axes)

    dlhs = _compute_gradient_for_operand(g, rhs, for_dlhs=True, target=lhs_in)
    drhs = _compute_gradient_for_operand(g, lhs, for_dlhs=False, target=rhs_in)

    if lhs_calibration is not None:
        dlhs = qarray.clip_gradient_to_calibration(dlhs, lhs_in, lhs_calibration, config.lhs_calibration_method)
    if rhs_calibration is not None:
        drhs = qarray.clip_gradient_to_calibration(drhs, rhs_in, rhs_calibration, config.rhs_calibration_method)

    return dlhs, drhs, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def _dot_general_qt_fwd_bwd(
    lhs,
    rhs,
    lhs_calibration,
    rhs_calibration,
    dimension_numbers,
    config,
    out_sharding,
):
    result, _ = _dot_general_qt_fwd(
        lhs,
        rhs,
        lhs_calibration,
        rhs_calibration,
        dimension_numbers,
        config,
        out_sharding,
    )
    return result


_dot_general_qt_fwd_bwd.defvjp(_dot_general_qt_fwd, _dot_general_qt_bwd)


def _dot_general_qt(lhs, rhs, dimension_numbers, config, out_sharding=None):
    lhs_calibration = None
    rhs_calibration = None

    if config.lhs_qtype and numerics.should_quantize(lhs.dtype):
        lhs_how = qwix_dot_general.get_how_to_quantize(
            dimension_numbers=dimension_numbers,
            ndims=(lhs.ndim, rhs.ndim),
            for_lhs=True,
            qtype=config.lhs_qtype,
            tile_size=config.tile_size,
            calibration_method=config.lhs_calibration_method,
        )
        if config.lhs_disable_channelwise_axes:
            lhs_how = dataclasses.replace(lhs_how, channelwise_axes=[])
        lhs_calibration = qarray.calibrate(lhs, lhs_how)
        if config.lhs_collect_quant_stat:
            lhs_calibration = config.lhs_collect_quant_stat(lhs_calibration)

    if config.rhs_qtype and numerics.should_quantize(rhs.dtype):
        rhs_how = qwix_dot_general.get_how_to_quantize(
            dimension_numbers=dimension_numbers,
            ndims=(lhs.ndim, rhs.ndim),
            for_lhs=False,
            qtype=config.rhs_qtype,
            tile_size=config.tile_size,
            calibration_method=config.rhs_calibration_method,
        )
        if config.rhs_disable_channelwise_axes:
            rhs_how = dataclasses.replace(rhs_how, channelwise_axes=[])
        rhs_calibration = qarray.calibrate(rhs, rhs_how)
        if config.rhs_collect_quant_stat:
            rhs_calibration = config.rhs_collect_quant_stat(rhs_calibration)

    if out_sharding is None:
        out_sharding = _inferred_out_sharding(lhs, rhs, dimension_numbers)

    return _dot_general_qt_fwd_bwd(
        lhs,
        rhs,
        lhs_calibration,
        rhs_calibration,
        dimension_numbers,
        config,
        out_sharding,
    )


class ExplicitQtProvider(QtProvider):
    def dot_general(
        self,
        lhs,
        rhs,
        dimension_numbers,
        precision=None,
        preferred_element_type=None,
        *,
        out_sharding=None,
    ):
        rule, op_id = self._get_current_rule_and_op_id("dot_general")
        if rule is None or rule.weight_qtype is None:
            return jax.lax.dot_general(
                lhs,
                rhs,
                dimension_numbers,
                precision=precision,
                preferred_element_type=preferred_element_type,
                out_sharding=out_sharding,
            )
        config = self._create_dot_general_qt_config(rule, op_id, lhs, rhs)
        return _dot_general_qt(lhs, rhs, dimension_numbers, config, out_sharding=out_sharding)


def quantize_model(model, rules, *sample_inputs, **sample_kwargs):
    return qwix.quantize_model(model, ExplicitQtProvider(rules), *sample_inputs, **sample_kwargs)


def int8_rules(module_path=".*layers.*"):
    return [
        qwix.QtRule(
            module_path=module_path,
            weight_qtype=jnp.int8,
            act_qtype=jnp.int8,
            bwd_qtype=jnp.int8,
            op_names=("dot_general",),
        )
    ]


def fp8_rules(module_path=".*layers.*"):
    return [
        qwix.QtRule(
            module_path=module_path,
            weight_qtype=jnp.float8_e4m3fn,
            act_qtype=jnp.float8_e4m3fn,
            bwd_qtype=jnp.float8_e5m2,
            op_names=("dot_general",),
        )
    ]


def maybe_quantize(model, quantization, *sample_inputs, **sample_kwargs):
    match quantization:
        case "int8":
            return quantize_model(model, int8_rules(), *sample_inputs, **sample_kwargs)
        case "fp8":
            return quantize_model(model, fp8_rules(), *sample_inputs, **sample_kwargs)
        case _:
            return model
