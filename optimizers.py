# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
Implements a bounds-aware Adam optimizer and utility loss functions
(L1, L2, PSNR) for optimization tasks using Dr.Jit and Mitsuba.
'''

from __future__ import annotations # Delayed parsing of type annotations

from contextlib import contextmanager
from collections import defaultdict
import drjit as dr
import mitsuba as mi
import numpy as np
from typing import (
    Any,
    Generic,
    Optional,
    Union,
    Mapping,
    MutableMapping,
    Iterator,
    Type,
    Tuple,
    Dict,
    TypeVar,
)
import sys

Extra = TypeVar("Extra")

class OldOptimizer:
    """
    Base class of all gradient-based optimizers.
    """
    def __init__(self, lr, params: dict):
        """
        Parameter ``lr``:
            learning rate

        Parameter ``params`` (:py:class:`dict`):
            Dictionary-like object containing parameters to optimize.
        """
        self.lr   = defaultdict(lambda: self.lr_default)
        self.lr_v = defaultdict(lambda: self.lr_default_v)

        self.set_learning_rate(lr)
        self.variables = {}
        self.state = {}

        if params is not None:
            for k, v in params.items():
                if type(params) is mi.SceneParameters:
                    if params.flags(k) & mi.ParamFlags.NonDifferentiable.value:
                        continue
                self.__setitem__(k, v)

    def __contains__(self, key: str):
        return self.variables.__contains__(key)

    def __getitem__(self, key: str):
        return self.variables[key]

    def __setitem__(self, key: str, value):
        """
        Overwrite the value of a parameter.

        This method can also be used to load a scene parameters to optimize.

        Note that the state of the optimizer (e.g. momentum) associated with the
        parameter is preserved, unless the parameter's dimensions changed. When
        assigning a substantially different parameter value (e.g. as part of a
        different optimization run), the previous momentum value is meaningless
        and calling `reset()` is advisable.
        """
        if not (dr.is_diff_v(value) and dr.is_float_v(value)):
            raise Exception('Optimizer.__setitem__(): value should be differentiable!')
        needs_reset = (key not in self.variables) or dr.shape(self.variables[key]) != dr.shape(value)

        self.variables[key] = dr.detach(value, True)
        dr.enable_grad(self.variables[key])
        if needs_reset:
            self.reset(key)

    def __delitem__(self, key: str) -> None:
        del self.variables[key]

    def __len__(self) -> int:
        return len(self.variables)

    def keys(self):
        return self.variables.keys()

    def items(self):
        class OptimizerItemIterator:
            def __init__(self, items):
                self.items = items
                self.it = items.keys().__iter__()

            def __iter__(self):
                return self

            def __next__(self):
                key = next(self.it)
                return (key, self.items[key])

        return OptimizerItemIterator(self.variables)

    def set_learning_rate(self, lr) -> None:
        """
        Set the learning rate.

        Parameter ``lr`` (``float``, ``dict``):
            The new learning rate. A ``dict`` can be provided instead to
            specify the learning rate for specific parameters.
        """

        # We use `dr.opaque` so the that the JIT compiler does not include
        # the learning rate as a scalar literal into generated code, which
        # would defeat kernel caching when updating learning rates.
        if isinstance(lr, float) or isinstance(lr, int):
            self.lr_default = float(lr)
            self.lr_default_v = dr.opaque(dr.detached_t(mi.Float), lr, shape=1)
        elif isinstance(lr, dict):
            for k, v in lr.items():
                self.lr[k] = v
                self.lr_v[k] = dr.opaque(dr.detached_t(mi.Float), v, shape=1)
        else:
            raise Exception('Optimizer.set_learning_rate(): value should be a float or a dict!')

    def reset(self, key):
        """
        Resets the internal state associated with a parameter, if any (e.g. momentum).
        """
        pass


class BoundedAdam(OldOptimizer):
    '''
    Implements a bounds-aware Adam optimizer.

    The BoundedAdam class allows for setting bounds on each parameter. If a
    gradient step reaches one of the bounds, the optimized value will be moved
    by half of the distance towards the bound instead. This ensures that the
    optimization process can approach the bound as closely as possible without
    ever stepping over it. This usually works better than clamping as this
    process will also reset the optimizer state for this parameter.
    '''
    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 mask_updates=False, uniform=False, params: dict=None):
        '''
        Parameter ``lr``:
            learning rate

        Parameter ``beta_1``:
            controls the exponential averaging of first order gradient moments

        Parameter ``beta_2``:
            controls the exponential averaging of second order gradient moments

        Parameter ``mask_updates``:
            if enabled, parameters and state variables will only be updated in a
            given iteration if it received nonzero gradients in that iteration

        Parameter ``uniform``:
            if enabled, the optimizer will use the 'UniformAdam' variant of Adam
            [Nicolet et al. 2021], where the update rule uses the *maximum* of
            the second moment estimates at the current step instead of the
            per-element second moments.

        Parameter ``params`` (:py:class:`dict`):
            Optional dictionary-like object containing parameters to optimize.
        '''
        assert 0 <= beta_1 < 1 and 0 <= beta_2 < 1 and lr > 0 and epsilon > 0

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.mask_updates = mask_updates
        self.uniform = uniform
        self.t = defaultdict(lambda: 0)
        self.bounds = {}
        super().__init__(lr, params)

    def set_bounds(self, key, upper=None, lower=None):
        '''
        Set bounds for a parameter
        '''
        assert lower is None or upper is None or lower < upper, 'Upper bound should be higher than lower bound! Did you mix the argument order?'
        self.bounds[key] = (upper, lower)

    def step(self, active={}):
        '''
        Take a gradient step

        The active dictionary might contain masks for individual parameters.
        '''
        for k, p in self.variables.items():
            has_mask = k in active
            mask = active.get(k, mi.Bool(True))

            self.t[k] += 1
            lr_scale = dr.sqrt(1 - self.beta_2 ** self.t[k]) / (1 - self.beta_1 ** self.t[k])
            lr_scale = dr.opaque(dr.detached_t(mi.Float), lr_scale, shape=1)

            lr_t = self.lr_v[k] * lr_scale
            g_p = dr.grad(p)
            g_p = dr.select(dr.isnan(g_p), 0, g_p)
            shape = dr.shape(g_p)

            if shape == 0:
                continue
            elif shape != dr.shape(self.state[k][0]):
                # Reset state if data size has changed
                self.reset(k)

            m_tp, v_tp = self.state[k]
            m_t = self.beta_1 * m_tp + (1 - self.beta_1) * g_p
            v_t = self.beta_2 * v_tp + (1 - self.beta_2) * dr.square(g_p)

            if self.mask_updates:
                mask &= (g_p != 0.0)

            if self.mask_updates or has_mask:
                m_t = dr.select(mask, m_t, m_tp)
                v_t = dr.select(mask, v_t, v_tp)

            self.state[k] = (m_t, v_t)

            if self.uniform:
                step = lr_t * m_t / (dr.sqrt(dr.max(v_t)) + self.epsilon)
            else:
                step = lr_t * m_t / (dr.sqrt(v_t) + self.epsilon)

            if self.mask_updates or has_mask:
                step = dr.select(mask, step, 0.0)

            v = dr.detach(p)
            u = v - step
            u = type(p)(u)

            if k in self.bounds:
                upper, lower = self.bounds[k]
                over_boundary = mi.Bool(False)
                if upper is not None:
                    over_boundary = (u >= upper)
                    v = dr.select(over_boundary & (v >= upper), upper, v) # Make sure original value is in the bounds
                    u = dr.select(over_boundary, v + 0.5 * (upper - v), u)
                if lower is not None:
                    over_boundary = (u <= lower)
                    v = dr.select(over_boundary & (v <= lower), lower, v) # Make sure original value is in the bounds
                    u = dr.select(over_boundary, v - 0.5 * (v - lower), u)

                # Reset the momentum when the boundary was reached
                self.state[k] = (
                    dr.select(over_boundary, 0, self.state[k][0]),
                    dr.select(over_boundary, 0, self.state[k][1])
                )

            dr.enable_grad(u)
            self.variables[k] = u

            dr.schedule(self.state[k])
            dr.schedule(self.variables[k])

        dr.eval()

    def reset(self, key):
        '''
        Zero-initializes the internal state associated with a parameter
        '''
        p = self.variables[key]
        shape = dr.shape(p) if dr.is_tensor_v(p) else dr.width(p)
        self.state[key] = (dr.zeros(dr.detached_t(p), shape),
                           dr.zeros(dr.detached_t(p), shape))
        self.t[key] = 0

    def __repr__(self):
        return ('BoundedAdam[\n'
                '  variables = %s,\n'
                '  lr = %s,\n'
                '  betas = (%g, %g),\n'
                '  eps = %g\n'
                '  bounds = %s\n'
                ']' % (list(self.keys()), dict(self.lr, default=self.lr_default),
                       self.beta_1, self.beta_2, self.epsilon, self.bounds))