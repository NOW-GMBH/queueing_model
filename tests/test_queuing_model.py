import math
from typing import Dict, Any
import numpy as np
import pytest
from queuing_model.queuing_model import (
    queue_mgc_lee_longton,
    _compute_wq_for_lambda,
    _erlang_c_prob_wait,
    # Wrappers
    que_mgc,
    que_mgc_server_wq_qed,
    server_utilization,
    # Helpers
    _qed_servers,
    _auto_search_radius,
)
from pydantic import ValidationError


@pytest.fixture(scope="session")
def standard_params() -> Dict[str, Any]:
    """Standard Testparameter"""
    return {
        "lambda_target": 10.0,
        "charging_time_min": 45,
        "stdev_ct_min": 10,
        "waiting_times_min": [5, 10, 20, 60],
        "method": "lee_longton",
        "beta": 1.0,
        "max_server": 20,
    }


class TestCoreFunctions:
    """Tests für die Kern‑Queuing‑Funktionen"""

    def test_compute_wq_for_lambda_stability(self):
        """Edge Cases: Instabilität, λ=0, c=1"""
        mu, charging_time, vk, ca_2 = 2.0, 0.5, 0.5, 1

        # Instabil: ρ >= 1
        roh, wq_mm, wq_mgc = _compute_wq_for_lambda(10, 4, mu, charging_time, vk, ca_2)
        assert not math.isfinite(wq_mgc)

        # λ = 0
        roh, wq_mm, wq_mgc = _compute_wq_for_lambda(0, 4, mu, charging_time, vk, ca_2)
        assert wq_mgc == 0.0

        # c=1: Pollaczek-Khinchine
        roh, wq_mm, wq_mgc = _compute_wq_for_lambda(1.5, 1, mu, charging_time, vk, ca_2)
        assert roh == pytest.approx(0.75, abs=1e-6)

    def test_erlang_c_prob_wait_correctness(self):
        """Bekannte Erlang-C Werte"""
        # M/M/1: P_wait = ρ
        assert _erlang_c_prob_wait(1, 0.5) == 0.5

        # ρ=0
        assert _erlang_c_prob_wait(5, 0.0) == 0.0

        # ρ=1
        assert _erlang_c_prob_wait(5, 1.0) == 1.0

    @pytest.mark.parametrize(
        "mean_waiting_time, servers",
        [
            (5.0 / 60, 10),  # strenges Ziel 5 Minuten in Stunden
            (1, 8),  # großzügig 1h
        ],
    )
    def test_queue_mgc_Adan_Resing_stable_convergence(
        self, mean_waiting_time: float, servers: int
    ):
        """Testet stabile Konvergenz"""
        mu = 60 / 45  # 1.333
        charging_time_h = 45 / 60
        vk = 10 / 45

        result = queue_mgc_lee_longton(
            mean_waiting_time, servers, mu, charging_time_h, vk
        )
        lambda_max, rho, wq_mm, wq_mgc, wz_az = result

        assert lambda_max > 0
        assert rho < 1.0
        assert wq_mgc <= mean_waiting_time * 1.01  # 1% Toleranz


class TestWrapperFunctions:
    """Tests für Wrapper und Integration"""

    @pytest.mark.parametrize(
        "wq_target, expected_min_servers",
        [
            (5.0, 10),
            (10.0, 9),
            (60.0, 8),
        ],
    )
    def test_que_mgc_server_wq_qed_correctness(
        self, standard_params, wq_target, expected_min_servers
    ):
        """Korrekte minimale Serverzahlen"""
        standard_params = standard_params.copy()
        standard_params["waiting_times_min"] = [wq_target]
        result = que_mgc_server_wq_qed(**standard_params)
        found_servers = result[1][str(wq_target)]
        assert found_servers == expected_min_servers

    def test_que_mgc_monotonic_increasing_lambda(self, standard_params):
        """que_mgc: λ_max(c) wächst monoton"""
        df = que_mgc(
            standard_params["charging_time_min"],
            standard_params["stdev_ct_min"],
            5.0,
            standard_params["max_server"],
            standard_params["method"],
        )
        lambdas = df["lambda"].values
        assert np.all(np.diff(lambdas) >= 0)  # Monoton wachsend

    def test_qed_servers_formula(self):
        """QED-Formel korrekt"""
        # Bekannte Werte
        assert _qed_servers(10, 2, 1.0) == 8  # ceil(5 + √5)
        assert _qed_servers(100, 3, 2.0) == 45  # ceil(33.3 + 2√33.3)

    def test_auto_search_radius_works(self):
        """Auto-Berechnung search_radius"""
        search_radius = _auto_search_radius(100, 3.0)
        assert search_radius == 12  # max(5,2√33.3)

    def test_search_radius_minimum(self):
        """Kleines System"""
        search_radius = _auto_search_radius(1, 6.0)
        assert search_radius == 5  # Minimum!


class TestConfigValidation:
    """Pydantic Validierung"""

    def test_invalid_method_raises(self):
        with pytest.raises(ValidationError, match="literal_error"):
            que_mgc_server_wq_qed(
                lambda_target_hours=10,
                charging_time_min=45,
                stdev_ct_min=10,
                waiting_times_min=[5],
                method="invalid_method",
                beta=1.0,
                search_radius=None,
                max_server=100,
            )

    def test_negative_lambda_error(self):
        with pytest.raises(ValidationError):
            que_mgc_server_wq_qed(
                lambda_target_hours=-1,
                ç=45,
                stdev_ct_min=10,
                waiting_times_min=[5],
                method="lee_longton",
                beta=1.0,
                search_radius=None,
                max_server=100,
            )


class TestEdgeCases:
    """Edge Cases und Robustheit"""

    def test_zero_waiting_time(self):
        """Wq=0 → unendlich viele Server oder None"""
        result = que_mgc_server_wq_qed(
            lambda_target_hours=10,
            charging_time_min=45,
            stdev_ct_min=10,
            waiting_times_min=[0.0],
            method="lee_longton",
        )
        assert result[1]["0.0"] is None or result[1]["0.0"] > 100

    def test_large_system_scaling(self):
        """Großes System testet Skalierung"""
        result = que_mgc_server_wq_qed(
            lambda_target_hours=1000,
            charging_time_min=45,
            stdev_ct_min=10,
            waiting_times_min=[5.0],
            method="lee_longton",
            beta=2,
            max_server=1000,
        )
        assert result[1]["5.0"] is not None
        assert result[1]["5.0"] < 1000


class TestPerformanceMetrics:
    """Zusätzliche Metriken"""

    def test_utilization_calculation(self):
        """ρ = λ/(cμ)"""
        rho = server_utilization(10, 10, 60 / 45)
        assert rho == pytest.approx(0.75, abs=1e-6)

    def test_erlang_c_plausibility(self):
        """P_wait monoton fallend mit c"""
        rhos = [_erlang_c_prob_wait(5, 0.8), _erlang_c_prob_wait(10, 0.8)]
        assert rhos[0] > rhos[1]


# pytest parametrized für alle Methoden
@pytest.mark.parametrize(
    "method,expected_servers",
    [
        ("lee_longton", {"5.0": 10, "10.0": 9, "20.0": 9, "60.0": 8}),
    ],
)
def test_que_mgc_server_wq_qed_consistency(standard_params, method, expected_servers):
    """que_mgc_server_wq_qed gibt erwartete Server-Anzahlen für alle Methoden"""
    params = standard_params.copy()
    params["method"] = method

    lambda_value, server_dict = que_mgc_server_wq_qed(**params)

    assert lambda_value > 0, f"{method}: Lambda sollte positiv sein"
    assert len(server_dict) == len(
        expected_servers
    ), f"{method}: Erwartet {len(expected_servers)} waiting_times, bekommen {len(server_dict)}"

    for waiting_time_str, expected_server in expected_servers.items():
        assert (
            waiting_time_str in server_dict
        ), f"{method}: waiting_time {waiting_time_str} fehlt in {server_dict}"
        assert (
            server_dict[waiting_time_str] == expected_server
        ), f"{method}: Für waiting_time={waiting_time_str} erwartet {expected_server}, bekommen {server_dict[waiting_time_str]}"


@pytest.mark.parametrize(
    "method,expected_servers",
    [
        ("lee_longton", {"5.0": 10, "10.0": 9, "20.0": 9, "60.0": 8}),
    ],
)
def test_que_mgc_server_wq_consistency(standard_params, method, expected_servers):
    """que_mgc_server_wq gibt erwartete Lambda- und Rho-Werte"""
    params = standard_params.copy()
    params = {k: v for k, v in params.items() if k != "beta"}
    params["method"] = method

    lambda_value, server_dict = que_mgc_server_wq_qed(**params)

    assert lambda_value > 0, f"{method}: Lambda sollte positiv sein"
    assert len(server_dict) == len(
        expected_servers
    ), f"{method}: Erwartet {len(expected_servers)} waiting_times, bekommen {len(server_dict)}"

    for waiting_time_str, expected_server in expected_servers.items():
        assert (
            waiting_time_str in server_dict
        ), f"{method}: waiting_time {waiting_time_str} fehlt in {server_dict}"
        assert (
            server_dict[waiting_time_str] == expected_server
        ), f"{method}: Für waiting_time={waiting_time_str} erwartet {expected_server}, bekommen {server_dict[waiting_time_str]}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
