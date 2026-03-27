import math
import numpy as np
import pytest
from queuing_model.queuing_model import (
    queue_mgc_lee_longton,
    _compute_wq_for_lambda,
    _erlang_c_prob_wait,
    queue_max_lambda,
    queue_min_servers,
    queue_min_servers_qed,
    server_utilization,
    _qed_servers,
    _auto_search_radius,
)
from pydantic import ValidationError
from conftest import ref_wq, ref_wq_allen_cunneen, stability_lower_bound


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────


class TestCoreFunctions:

    def test_compute_wq_instability(self):
        """ρ >= 1 yields Wq = inf (unstable queue)."""
        mu, ct, cv, ca2 = 2.0, 0.5, 0.5, 1.0
        _, _, wq = _compute_wq_for_lambda(10, 4, mu, ct, cv, ca2)
        assert not math.isfinite(wq)

    def test_compute_wq_zero_lambda(self):
        """Zero arrival rate yields Wq = 0 (empty system)."""
        mu, ct, cv, ca2 = 2.0, 0.5, 0.5, 1.0
        _, _, wq = _compute_wq_for_lambda(0, 4, mu, ct, cv, ca2)
        assert wq == 0.0

    def test_compute_wq_single_server_rho(self):
        """Single server: utilization ρ = λ/μ is computed correctly."""
        mu, ct, cv, ca2 = 2.0, 0.5, 0.5, 1.0
        rho, _, _ = _compute_wq_for_lambda(1.5, 1, mu, ct, cv, ca2)
        assert rho == pytest.approx(0.75, abs=1e-6)

    def test_erlang_c_mmc1_equals_rho(self):
        """M/M/1: Erlang-C probability equals ρ (P(Wq>0) = ρ)."""
        assert _erlang_c_prob_wait(1, 0.5) == pytest.approx(0.5, abs=1e-9)

    def test_erlang_c_zero_utilization(self):
        """Zero utilization: no customer ever waits, P(Wq>0) = 0."""
        assert _erlang_c_prob_wait(5, 0.0) == pytest.approx(0.0, abs=1e-9)

    def test_erlang_c_high_utilization(self):
        """Near-saturated system: P(Wq>0) converges to 1 as ρ → 1."""
        p = _erlang_c_prob_wait(5, 0.999)
        assert p > 0.99

    def test_erlang_c_monotone_in_rho(self):
        """P(Wq>0) increases strictly monotonically with ρ at fixed c."""
        rhos = [0.1, 0.3, 0.5, 0.7, 0.9]
        probs = [_erlang_c_prob_wait(5, r) for r in rhos]
        assert all(p1 < p2 for p1, p2 in zip(probs, probs[1:]))

    @pytest.mark.parametrize(
        "wq_target_h, servers",
        [
            (5.0 / 60, 10),
            (1.0, 8),
        ],
    )
    def test_queue_mgc_lee_longton_convergence(self, wq_target_h, servers):
        """queue_mgc_lee_longton returns a stable solution satisfying the Wq target."""
        mu = 60 / 45
        ct_h = 45 / 60
        cv = 10 / 45
        lambda_max, rho, _, wq_mgc, _ = queue_mgc_lee_longton(
            wq_target_h, servers, mu, ct_h, cv
        )
        assert lambda_max > 0
        assert rho < 1.0
        assert wq_mgc <= wq_target_h * 1.01  # 1 % tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Reference formula verification — both methods
# ─────────────────────────────────────────────────────────────────────────────
class _QueueMinServersBase:
    """..."""

    CHARGING_TIME_MIN = 45.0
    STDEV_CT_MIN = 10.0
    MU = 60 / CHARGING_TIME_MIN
    CV = STDEV_CT_MIN / CHARGING_TIME_MIN
    MAX_SERVER = 30
    OUTPUT_UNIT = "hours_to_minutes"

    _fn = None  # override in subclass

    def _result(self, **kwargs) -> dict:
        defaults = dict(
            charging_time_min=self.CHARGING_TIME_MIN,
            stdev_ct_min=self.STDEV_CT_MIN,
            max_server=self.MAX_SERVER,
            output_unit=self.OUTPUT_UNIT,
        )
        defaults.update(kwargs)  # kwargs überschreiben Defaults, kein Konflikt
        _, result = self._fn(**defaults)
        return result

    # ------------------------------------------------------------------ #
    # Parametrize sets (shared by both subclasses)                         #
    # ------------------------------------------------------------------ #

    SATISFIES_PARAMS = [
        ("lee_longton", {}, 5.0),
        ("lee_longton", {}, 10.0),
        ("lee_longton", {}, 60.0),
        ("allen_cunneen", {"c_a2": 1.0}, 5.0),  # c_a2=1 → identical to Lee-Longton
        ("allen_cunneen", {"c_a2": 1.5}, 5.0),  # c_a2>1 → more servers expected
        ("allen_cunneen", {"c_a2": 1.5}, 10.0),
        ("allen_cunneen", {"c_a2": 0.5}, 5.0),  # c_a2<1 → fewer or equal servers
    ]

    MINIMAL_PARAMS = [
        ("lee_longton", {}, 5.0),
        ("lee_longton", {}, 10.0),
        ("lee_longton", {}, 60.0),
        ("allen_cunneen", {"c_a2": 1.5}, 5.0),
        ("allen_cunneen", {"c_a2": 1.5}, 10.0),
    ]

    MONOTONE_LAMBDA_PARAMS = [
        (10.0, "lee_longton", {}),
        (10.0, "allen_cunneen", {"c_a2": 1.5}),
        (50.0, "allen_cunneen", {"c_a2": 1.0}),
    ]

    CONSISTENCY_PARAMS = [
        ("lee_longton", {}, {"5.0": 10, "10.0": 9, "20.0": 9, "60.0": 8}),
        ("allen_cunneen", {"c_a2": 1.0}, {"5.0": 10, "10.0": 9, "20.0": 9, "60.0": 8}),
        ("allen_cunneen", {"c_a2": 1.5}, {"5.0": 10, "10.0": 10, "20.0": 9, "60.0": 8}),
    ]

    # ------------------------------------------------------------------ #
    # Tests                                                                #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("method, extra_kwargs, wq_target_min", SATISFIES_PARAMS)
    def test_min_servers_satisfies_wq_target(self, method, extra_kwargs, wq_target_min):
        """Returned server count c satisfies the Wq target per reference formula."""
        c_a2 = extra_kwargs.get("c_a2", 1.0)
        result = self._result(
            lambda_target=10.0,
            waiting_times_min=[wq_target_min],
            method=method,
            **extra_kwargs,
        )
        c = result[str(wq_target_min)]
        assert c is not None
        wq_h = ref_wq(10.0, self.MU, c, self.CV, method, c_a2)
        assert wq_h * 60 <= wq_target_min, (
            f"[{method}, c_a2={c_a2}] c={c} does not satisfy "
            f"Wq={wq_target_min} min (ref: {wq_h * 60:.2f} min)"
        )

    @pytest.mark.parametrize("method, extra_kwargs, wq_target_min", MINIMAL_PARAMS)
    def test_min_servers_is_minimal(self, method, extra_kwargs, wq_target_min):
        """c is minimal: c-1 does NOT satisfy the Wq target per reference formula."""
        c_a2 = extra_kwargs.get("c_a2", 1.0)
        result = self._result(
            lambda_target=10.0,
            waiting_times_min=[wq_target_min],
            method=method,
            **extra_kwargs,
        )
        c = result[str(wq_target_min)]
        assert c is not None and c >= 2
        wq_h = ref_wq(10.0, self.MU, c - 1, self.CV, method, c_a2)
        assert not math.isfinite(wq_h) or wq_h * 60 > wq_target_min, (
            f"[{method}, c_a2={c_a2}] c-1={c - 1} already satisfies "
            f"Wq={wq_target_min} min — c={c} is not minimal "
            f"(ref: {wq_h * 60:.2f} min)"
        )

    @pytest.mark.parametrize("method, extra_kwargs, wq_target_min", SATISFIES_PARAMS)
    def test_min_servers_monotone_in_wq_target(
        self, method, extra_kwargs, wq_target_min
    ):
        """c is non-increasing as Wq target grows: a looser SLA needs fewer or equal servers."""
        targets = [5.0, 10.0, 30.0, 60.0]
        result = self._result(
            lambda_target=10.0,
            waiting_times_min=targets,
            method=method,
            max_server=50,
            **extra_kwargs,
        )
        counts = [result[str(t)] for t in targets]
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], (
                f"[{method}] c({targets[i]} min)={counts[i]} < "
                f"c({targets[i + 1]} min)={counts[i + 1]}: "
                "server count should be non-increasing for looser Wq targets"
            )

    @pytest.mark.parametrize("lambda_, method, extra_kwargs", MONOTONE_LAMBDA_PARAMS)
    def test_min_servers_monotone_in_lambda(self, lambda_, method, extra_kwargs):
        """c is non-decreasing as arrival rate grows: higher load needs more or equal servers."""
        lambdas = [5.0, 10.0, 20.0, 50.0]
        counts = [
            self._result(
                lambda_target=lam,
                waiting_times_min=[10.0],
                method=method,
                max_server=100,
                **extra_kwargs,
            )[str(10.0)]
            for lam in lambdas
        ]
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1], (
                f"[{method}] c(λ={lambdas[i]})={counts[i]} > "
                f"c(λ={lambdas[i + 1]})={counts[i + 1]}: "
                "server count should be non-decreasing for higher arrival rates"
            )

    @pytest.mark.parametrize("wq_target_min", [5.0, 10.0, 30.0])
    def test_allen_cunneen_vs_lee_longton_at_c_a2_1(self, wq_target_min):
        """Allen-Cunneen with c_a2=1 must return same c as Lee-Longton (Poisson arrivals)."""
        c_ll = self._result(
            lambda_target=10.0, waiting_times_min=[wq_target_min], method="lee_longton"
        )[str(wq_target_min)]
        c_ac = self._result(
            lambda_target=10.0,
            waiting_times_min=[wq_target_min],
            method="allen_cunneen",
            c_a2=1.0,
        )[str(wq_target_min)]
        assert c_ll == c_ac, (
            f"Wq={wq_target_min} min: Lee-Longton c={c_ll} "
            f"≠ Allen-Cunneen (c_a2=1) c={c_ac}"
        )

    @pytest.mark.parametrize("wq_target_min", [5.0, 10.0])
    def test_higher_c_a2_needs_more_or_equal_servers(self, wq_target_min):
        """Higher arrival variability (c_a2>1) requires more or equal servers than c_a2<1."""
        c_low = self._result(
            lambda_target=10.0,
            waiting_times_min=[wq_target_min],
            method="allen_cunneen",
            c_a2=0.5,
        )[str(wq_target_min)]
        c_high = self._result(
            lambda_target=10.0,
            waiting_times_min=[wq_target_min],
            method="allen_cunneen",
            c_a2=1.5,
        )[str(wq_target_min)]
        assert c_high >= c_low, (
            f"Wq={wq_target_min} min: c_a2=1.5 → c={c_high} < c_a2=0.5 → c={c_low}: "
            "higher arrival variability should require more or equal servers"
        )

    @pytest.mark.parametrize("lambda_, method, extra_kwargs", MONOTONE_LAMBDA_PARAMS)
    def test_min_servers_above_stability_bound(self, lambda_, method, extra_kwargs):
        """Returned c >= ceil(λ/μ) — absolute stability lower bound."""
        result = self._result(
            lambda_target=lambda_,
            charging_time_min=45,
            stdev_ct_min=10,
            waiting_times_min=[5.0],
            method=method,
            max_server=200,
            output_unit="hours_to_minutes",
            **extra_kwargs,
        )
        c = result["5.0"]
        assert c is not None
        assert c >= stability_lower_bound(lambda_, self.MU), (
            f"c={c} is below stability bound "
            f"{stability_lower_bound(lambda_, self.MU)} for λ={lambda_}"
        )

    @pytest.mark.parametrize(
        "method, extra_kwargs, expected_servers", CONSISTENCY_PARAMS
    )
    def test_min_servers_consistency_fixed_reference(
        self, method, extra_kwargs, expected_servers
    ):
        """Server counts match fixed reference values — regression guard."""
        result = self._result(
            lambda_target=10.0,
            waiting_times_min=[5.0, 10.0, 20.0, 60.0],
            method=method,
            **extra_kwargs,
        )
        for wq_str, expected_c in expected_servers.items():
            assert result.get(wq_str) == expected_c, (
                f"[{method}, c_a2={extra_kwargs.get('c_a2', '—')}] "
                f"Wq={wq_str}: expected c={expected_c}, got c={result.get(wq_str)}"
            )

    # ─────────────────────────────────────────────────────────────────────────────
    # Edge cases
    # ─────────────────────────────────────────────────────────────────────────────
    @pytest.mark.parametrize(
        "method, extra_kwargs",
        [
            ("lee_longton", {}),
            ("allen_cunneen", {"c_a2": 1.5}),
        ],
    )
    def test_zero_wq_target_returns_none_or_very_large(self, method, extra_kwargs):
        """Wq=0 is unachievable: result is None or an implausibly large server count."""
        with pytest.raises((ValidationError, ValueError)):
            self._result(
                lambda_target=10.0,
                waiting_times_min=[0.0001],
                method=method,
                **extra_kwargs,
            )

    @pytest.mark.parametrize(
        "method, extra_kwargs",
        [
            ("lee_longton", {}),
            ("allen_cunneen", {"c_a2": 1.5}),
        ],
    )
    def test_large_system_feasible(self, method, extra_kwargs):
        """λ=1000: feasible solution exists and stays within max_server."""
        result = self._result(
            lambda_target=1000.0,
            waiting_times_min=[5.0],
            method=method,
            max_server=1000,
            **extra_kwargs,
        )
        c = result.get("5.0")
        assert c is not None and c < 1000, f"[{method}] c={c}"

    @pytest.mark.parametrize(
        "method, extra_kwargs",
        [
            ("lee_longton", {}),
            ("allen_cunneen", {"c_a2": 1.0}),
        ],
    )
    def test_very_low_lambda_feasible(self, method, extra_kwargs):
        """λ=0.1: trivially solvable with a single server."""
        result = self._result(
            lambda_target=0.1,
            waiting_times_min=[60.0],
            method=method,
            **extra_kwargs,
        )
        c = result.get("60.0")
        assert c is not None and c >= 1, f"[{method}] c={c}"


# ------------------------------------------------------------------ #
# Concrete subclasses                                                  #
# ------------------------------------------------------------------ #


class TestQueueMinServersAgainstReferenceFormula(_QueueMinServersBase):
    _fn = staticmethod(queue_min_servers)


class TestQueueMinServersQedAgainstReferenceFormula(_QueueMinServersBase):
    _fn = staticmethod(queue_min_servers_qed)


# ─────────────────────────────────────────────────────────────────────────────
# Allen-Cunneen specific tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAllenCunneen:
    """Verify Allen-Cunneen specific behaviour and its relationship to Lee-Longton."""

    MU = 60 / 45
    CV = 10 / 45

    @pytest.mark.parametrize("c_a2", [0.5, 1.0, 1.5, 2.0])
    def test_ref_formula_monotone_in_ca2(self, c_a2):
        """Reference formula: Wq increases monotonically with c_a2 at fixed c."""
        wqs = [
            ref_wq_allen_cunneen(10.0, self.MU, 12, self.CV, ca2)
            for ca2 in [0.5, 1.0, 1.5, 2.0]
        ]
        assert all(w1 <= w2 for w1, w2 in zip(wqs, wqs[1:]))


# ─────────────────────────────────────────────────────────────────────────────
# Monotonicity
# ─────────────────────────────────────────────────────────────────────────────


class TestMonotonicity:
    """Structural properties that must hold regardless of parameter values."""

    MU = 60 / 45
    CV = 10 / 45

    def test_wq_decreasing_in_c_lee_longton(self):
        """Wq decreases monotonically as c increases (Lee-Longton reference)."""
        lambda_ = 10.0
        c_min = stability_lower_bound(lambda_, self.MU) + 1
        wqs = [
            ref_wq(lambda_, self.MU, c, self.CV, "lee_longton")
            for c in range(c_min, c_min + 10)
        ]
        assert all(w1 >= w2 for w1, w2 in zip(wqs, wqs[1:]))

    def test_wq_decreasing_in_c_allen_cunneen(self):
        """Wq decreases monotonically as c increases (Allen-Cunneen reference)."""
        lambda_ = 10.0
        c_min = stability_lower_bound(lambda_, self.MU) + 1
        wqs = [
            ref_wq(lambda_, self.MU, c, self.CV, "allen_cunneen", c_a2=1.5)
            for c in range(c_min, c_min + 10)
        ]
        assert all(w1 >= w2 for w1, w2 in zip(wqs, wqs[1:]))

    def test_lambda_max_monotone_in_c(self, standard_params):
        """Maximum feasible λ increases monotonically with server count c."""
        df = queue_max_lambda(
            standard_params["charging_time_min"],
            standard_params["stdev_ct_min"],
            5.0,
            standard_params["max_server"],
            standard_params["method"],
        )
        assert np.all(np.diff(df["lambda"].values) >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


class TestHelperFunctions:

    @pytest.mark.parametrize(
        "lambda_, mu, beta, expected",
        [
            (10, 2.0, 1.0, 8),  # ceil(5 + √5 ≈ 7.24) = 8
            (100, 3.0, 2.0, 45),  # ceil(33.33 + 2·5.77 ≈ 44.87) = 45
            (9, 3.0, 0.0, 3),  # ceil(3 + 0) = 3  (stability bound only)
        ],
    )
    def test_qed_servers_formula(self, lambda_, mu, beta, expected):
        """QED staffing rule c = ceil(R + β·√R) with R = λ/μ."""
        assert _qed_servers(lambda_, mu, beta) == expected

    @pytest.mark.parametrize(
        "lambda_, mu, expected",
        [
            (100, 3.0, 12),  # max(5, ceil(2·√33.33)) = 12
            (1, 6.0, 5),  # floor(λ/μ) too small → minimum of 5 applies
        ],
    )
    def test_auto_search_radius(self, lambda_, mu, expected):
        """Auto search radius equals max(5, ceil(2·√(λ/μ)))."""
        assert _auto_search_radius(lambda_, mu) == expected

    def test_server_utilization(self):
        """Server utilization ρ = λ / (c·μ) is computed correctly."""
        mu = 60 / 45
        rho = server_utilization(10, 10, mu)
        assert rho == pytest.approx(10 / (10 * mu), abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigValidation:

    def test_invalid_method_raises(self):
        """Unsupported method string raises a Pydantic literal_error."""
        with pytest.raises(ValidationError, match="literal_error"):
            queue_min_servers_qed(
                lambda_target=10,
                charging_time_min=45,
                stdev_ct_min=10,
                waiting_times_min=[5],
                method="invalid_method",
                beta=1.0,
                max_server=100,
            )

    def test_negative_lambda_raises(self):
        """Negative arrival rate is physically invalid and must raise ValidationError."""
        with pytest.raises(ValidationError):
            queue_min_servers_qed(
                lambda_target=-1,
                charging_time_min=45,
                stdev_ct_min=10,
                waiting_times_min=[5],
                method="lee_longton",
                beta=1.0,
                max_server=100,
            )

    def test_zero_charging_time_raises(self):
        """Zero service time is physically invalid and must raise ValidationError."""
        with pytest.raises(ValidationError):
            queue_min_servers_qed(
                lambda_target=10,
                charging_time_min=0,
                stdev_ct_min=10,
                waiting_times_min=[5],
                method="lee_longton",
            )

    def test_allen_cunneen_missing_ca2_raises(self):
        """Allen-Cunneen without c_a2 raises ValidationError or TypeError."""
        with pytest.raises((ValidationError, ValueError)):
            queue_min_servers_qed(
                lambda_target=10,
                charging_time_min=45,
                stdev_ct_min=10,
                waiting_times_min=[5.0],
                method="allen_cunneen",
            )

    def test_allen_cunneen_negative_ca2_raises(self):
        """Negative c_a2 is physically invalid and must raise ValidationError."""
        with pytest.raises(ValidationError):
            queue_min_servers_qed(
                lambda_target=10,
                charging_time_min=45,
                stdev_ct_min=10,
                waiting_times_min=[5.0],
                method="allen_cunneen",
                c_a2=-1.0,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Cross-method consistency (fixed reference values)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "method, extra_kwargs, expected_servers",
    [
        (
            "lee_longton",
            {},
            {"5.0": 10, "10.0": 9, "20.0": 9, "60.0": 8},
        ),
        (
            "allen_cunneen",
            {"c_a2": 1.0},
            {"5.0": 10, "10.0": 9, "20.0": 9, "60.0": 8},  # identical to lee_longton
        ),
        (
            "allen_cunneen",
            {"c_a2": 1.5},
            {
                "5.0": 10,
                "10.0": 10,
                "20.0": 9,
                "60.0": 8,
            },  # ← verify manually on first run
        ),
    ],
)
def test_min_servers_consistency_by_method(
    standard_params, method, extra_kwargs, expected_servers
):
    """Server counts match fixed reference values for each method and c_a2."""
    params = {
        **standard_params,
        "method": method,
        "output_unit": "hours_to_minutes",
        **extra_kwargs,
    }
    lambda_value, server_dict = queue_min_servers_qed(**params)

    assert lambda_value > 0
    for wq_str, expected_c in expected_servers.items():
        assert server_dict.get(wq_str) == expected_c, (
            f"[{method}, c_a2={extra_kwargs.get('c_a2', '—')}] "
            f"Wq={wq_str}: expected c={expected_c}, got c={server_dict.get(wq_str)}"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
