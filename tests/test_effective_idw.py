#!/usr/bin/env python3
"""Math-kernel tests for rqab.effective_idw (+ the w-table interpolator).

Port of the pre-refactor tests/test_effective_idw.py; the golden numeric
anchors are unchanged.  The kernel moved from scripts/effective_idw.py to
rqab/effective_idw.py and WTableInterpolator from scripts/plot_w_overlay_k.py
to rqab/tables.py.
"""
from __future__ import annotations

import math
import unittest

import helpers  # noqa: F401  (sys.path bootstrap)

from rqab.effective_idw import (
    distribution_beta_at_zero,
    distribution_moments,
    effective_idw_approx,
    h2_shape_from_canonical,
    h2_params_from_rate_scv_r,
    hat_idw_refined,
    hyperexponential2_moments,
    idc_erlang_equilibrium,
    idc_h2_equilibrium,
    tau_tilde_c,
)
from rqab.tables import WTableInterpolator


class TestH2Parameters(unittest.TestCase):
    def test_roundtrip_rate_scv_r(self) -> None:
        rate = 1.3
        scv = 4.0
        r = 0.5

        params = h2_params_from_rate_scv_r(rate=rate, scv=scv, r=r)
        self.assertGreaterEqual(params.mu1, params.mu2)

        mean = params.p / params.mu1 + (1.0 - params.p) / params.mu2
        rec_rate = 1.0 / mean
        second = 2.0 * (
            params.p / (params.mu1 * params.mu1)
            + (1.0 - params.p) / (params.mu2 * params.mu2)
        )
        var = second - mean * mean
        rec_scv = var / (mean * mean)
        rec_r = (params.p / params.mu1) / mean

        self.assertAlmostEqual(rec_rate, rate, places=11)
        self.assertAlmostEqual(rec_scv, scv, places=10)
        self.assertAlmostEqual(rec_r, r, places=11)

    def test_default_balanced_mean(self) -> None:
        a = h2_params_from_rate_scv_r(rate=2.0, scv=4.0)
        b = h2_params_from_rate_scv_r(rate=2.0, scv=4.0, r=0.5)
        self.assertAlmostEqual(a.p, b.p, places=13)
        self.assertAlmostEqual(a.mu1, b.mu1, places=13)
        self.assertAlmostEqual(a.mu2, b.mu2, places=13)

    def test_degenerate_scv_one(self) -> None:
        params = h2_params_from_rate_scv_r(rate=2.0, scv=1.0, r=0.4)
        self.assertAlmostEqual(params.mu1, 2.0, places=12)
        self.assertAlmostEqual(params.mu2, 2.0, places=12)
        self.assertAlmostEqual(params.beta, 0.0, places=12)

        values = idc_h2_equilibrium([0.0, 0.3, 10.0], rate=2.0, scv=1.0, r=0.4)
        for val in values:
            self.assertAlmostEqual(val, 1.0, places=12)


class TestIDCAndApproximation(unittest.TestCase):
    def test_idc_limits(self) -> None:
        idc0 = idc_h2_equilibrium(0.0, rate=1.0, scv=4.0, r=0.5)
        idc_large = idc_h2_equilibrium(1e8, rate=1.0, scv=4.0, r=0.5)
        self.assertAlmostEqual(idc0, 1.0, places=13)
        self.assertAlmostEqual(idc_large, 4.0, places=6)

    def test_effective_idw_identity_when_w_is_one(self) -> None:
        t = [1.0, 2.0, 3.0]
        ia_t = [1.2, 1.8, 2.2]
        rho = 1.3
        c_s2 = 1.0
        hat = hat_idw_refined(t=t, ia_t=ia_t, rho=rho, c_s2=c_s2)
        approx = effective_idw_approx(
            t=t,
            ia_t=ia_t,
            rho=rho,
            c_s2=c_s2,
            w_values=[1.0, 1.0, 1.0],
        )
        self.assertEqual(len(hat), len(approx))
        for lhs, rhs in zip(hat, approx):
            self.assertAlmostEqual(lhs, rhs, places=13)

    def test_erlang_idc_limits(self) -> None:
        idc0 = idc_erlang_equilibrium(0.0, k=2, rate=2.0)
        idc_large = idc_erlang_equilibrium(1e8, k=2, rate=2.0)
        self.assertAlmostEqual(idc0, 1.0, places=13)
        self.assertAlmostEqual(idc_large, 0.5, places=6)

    def test_erlang_k2_closed_form(self) -> None:
        # For k=2, IDC(t) = 1/2 + (1 - exp(-2*rate*t)) / (4*rate*t)
        t = 0.7
        rate = 1.8
        got = idc_erlang_equilibrium(t, k=2, rate=rate)
        expected = 0.5 + (1.0 - math.exp(-2.0 * rate * t)) / (4.0 * rate * t)
        self.assertAlmostEqual(got, expected, places=12)

    def test_distribution_moments_helpers(self) -> None:
        mean_e, scv_e = distribution_moments("exponential", {"rate": 2.0})
        self.assertAlmostEqual(mean_e, 0.5, places=12)
        self.assertAlmostEqual(scv_e, 1.0, places=12)

        mean_er, scv_er = distribution_moments("erlang_k", {"k": 4, "rate": 2.0})
        self.assertAlmostEqual(mean_er, 2.0, places=12)
        self.assertAlmostEqual(scv_er, 0.25, places=12)

        mean_ln, scv_ln = distribution_moments("lognormal", {"mean": 1.0, "scv": 4.0})
        self.assertAlmostEqual(mean_ln, 1.0, places=12)
        self.assertAlmostEqual(scv_ln, 4.0, places=12)

        mean_h2, scv_h2 = hyperexponential2_moments(0.5, 3.0, 1.0 / 3.0)
        self.assertGreater(mean_h2, 0.0)
        self.assertGreater(scv_h2, 1.0)

        shape_scv, shape_r = h2_shape_from_canonical(0.5, 3.0, 1.0 / 3.0)
        self.assertAlmostEqual(shape_scv, scv_h2, places=12)
        self.assertTrue(0.0 < shape_r < 1.0)

    def test_distribution_beta_at_zero_helpers(self) -> None:
        beta_exp = distribution_beta_at_zero("exponential", {"rate": 1.7}, k=1)
        self.assertAlmostEqual(beta_exp, 1.7, places=12)

        beta_er2 = distribution_beta_at_zero("erlang_k", {"k": 2, "rate": 2.0}, k=2)
        self.assertAlmostEqual(beta_er2, 2.0, places=12)

        beta_h2 = distribution_beta_at_zero(
            "hyperexponential2",
            {
                "p": 0.8872983346207416,
                "rate1": 1.7745966692414832,
                "rate2": 0.22540333075851682,
            },
            k=1,
        )
        self.assertAlmostEqual(beta_h2, 1.6, places=12)

    def test_tau_tilde_c_for_target_model(self) -> None:
        tau, tilde_c = tau_tilde_c(
            c=2.0,
            k=2,
            mu=1.0,
            c_a2=4.0,
            c_s2=1.0,
            beta_patience=2.0,
        )
        self.assertAlmostEqual(tau, 10.0 ** (1.0 / 3.0), places=12)
        expected_tilde = 2.0 * (2.5 ** (-2.0 / 3.0)) * (2.0 ** (-1.0 / 3.0))
        self.assertAlmostEqual(tilde_c, expected_tilde, places=12)


class TestWTableInterpolator(unittest.TestCase):
    def test_no_hard_shape_enforcement(self) -> None:
        interp = WTableInterpolator(
            c_grid=[0.0, 1.0],
            t_grid=[0.0, 1.0, 2.0],
            w_matrix=[
                [1.0, 1.2, 0.9],
                [1.0, 1.4, 1.6],
            ],
        )
        self.assertAlmostEqual(interp.w(0.0, 1.0), 1.2, places=12)
        self.assertGreater(interp.w(1.0, 2.0), 1.0)

        curve = interp.curve(1.0, [1.0, 2.0])
        self.assertGreater(curve[1], curve[0])

    def test_t0_anchor(self) -> None:
        interp = WTableInterpolator(
            c_grid=[-1.0, 0.0],
            t_grid=[0.5, 2.0],
            w_matrix=[
                [0.2, 0.3],
                [0.4, 0.1],
            ],
        )
        self.assertAlmostEqual(interp.w(-0.5, 0.0), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
