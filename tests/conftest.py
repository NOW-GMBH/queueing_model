# tests/conftest.py

import math
import pytest
from typing import Dict, Any


def _ref_erlang_c(c: int, rho: float) -> float:
    """Erlang-C probability P(Wq > 0) — probability that an arriving customer has to wait.

    Erlang-C probability P(Wq > 0):

    C(c, a) = (a^c / c!) · 1/(1-ρ)
              ─────────────────────────────────────────
              Σ_{k=0}^{c-1} (a^k / k!) + (a^c / c!) · 1/(1-ρ)

    where   a = c·ρ = λ/μ  (offered load in Erlang)
            ρ = λ/(c·μ)    (server utilization, ρ < 1)

    Parameters
    ----------
    c : int
        Number of servers.
    rho : float
        Server utilization ρ = λ / (c·μ). Must satisfy 0 ≤ ρ < 1.

    Returns
    -------
    float
        P(Wq > 0) ∈ [0, 1).

    References
    ----------
    Erlang, A. K. (1917). Solution of Some Problems in the Theory of Probabilities
        of Significance in Automatic Telephone Exchanges.
        Post Office Electrical Engineers Journal, 10, 189–197.
    Cooper, R.B. (1990). Queueing theory. In D.P. Heyman & M.J. Sobel (Eds.),
        Handbooks in Operations Research and Management Science (Vol. 2, pp. 469–518).
        Elsevier. https://doi.org/10.1016/S0927-0507(05)80174-4
    """

    if rho <= 0.0:
        return 0.0
    if rho >= 1.0:
        return 1.0
    a = c * rho
    sum_terms = sum(a**k / math.factorial(k) for k in range(c))
    last = (a**c) / (math.factorial(c) * (1 - rho))
    return last / (sum_terms + last)


def ref_wq_lee_longton(lambda_: float, mu: float, c: int, cv: float) -> float:
    """Mean waiting time Wq for an M/G/c queue (Lee & Longton approximation).

    Approximates Wq by scaling the M/M/c result with a service-time
    variability correction factor:

        Wq ≈ Wq_MMc · (1 + cv²) / 2

    where Wq_MMc is the M/M/c mean waiting time:

        Wq_MMc = C(c, a) / (c·μ - λ)

    and C(c, a) is the Erlang-C probability (see _ref_erlang_c),
    a = λ/μ the offered load, cv = σ/E[S] the coefficient of variation
    of the service time.

    Full expression:

        Wq(λ, μ, c, cv) = C(c, a) / (c·μ - λ) · (1 + cv²) / 2

    Parameters
    ----------
    lambda_ : float
        Arrival rate λ in vehicles per hour.
    mu : float
        Service rate μ = 1/E[S] in 1/hour.
    c : int
        Number of servers.
    cv : float
        Coefficient of variation of service time cv = σ/E[S]. cv=0
        corresponds to deterministic service (D), cv=1 to exponential (M).

    Returns
    -------
    float
        Mean waiting time Wq in hours. Returns 0.0 if λ=0, math.inf if ρ≥1.

    References
    ----------
    Khinchin, A. Y. (1932). Mathematical theory of a stationary queue.
        Matematicheskii Sbornik, 39(4), 73–84.
    Lee, A. M., Longton, P. A. (1959). Queueing processes associated
        with airline passenger check-in. Operational Research Quarterly,
        10, 56–71.
    Pollaczek, F. (1930). Über eine Aufgabe der Wahrscheinlichkeitstheorie. I
        Mathematische Zeitschrift, 32, 64–100.
    """
    if lambda_ == 0.0:
        return 0.0
    rho = lambda_ / (c * mu)
    if rho >= 1.0:
        return math.inf
    ec = _ref_erlang_c(c, rho)
    return (ec / (c * mu - lambda_)) * (1 + cv**2) / 2


def ref_wq_allen_cunneen(
    lambda_: float, mu: float, c: int, cv: float, c_a2: float
) -> float:
    """Mean waiting time Wq for a GI/G/c queue (Allen-Cunneen approximation).

    Generalises the Lee-Longton approximation to non-Poisson arrivals by
    replacing the service-time correction factor (1 + cv²)/2 with a term
    that also accounts for arrival-time variability:

        Wq ≈ Wq_MMc · (c_a2 + cv²) / 2

    where Wq_MMc is the M/M/c mean waiting time:

        Wq_MMc = C(c, a) / (c·μ - λ)

    Full expression:

        Wq(λ, μ, c, cv, c_a2) = C(c, a) / (c·μ - λ) · (c_a2 + cv²) / 2

    Relationship to Lee-Longton:
        Setting c_a2 = 1 (Poisson arrivals, i.e. M/G/c) recovers the
        Lee-Longton formula exactly.

    Parameters
    ----------
    lambda_ : float
        Arrival rate λ in vehicles per hour.
    mu : float
        Service rate μ = 1/E[S] in 1/hour.
    c : int
        Number of servers.
    cv : float
        Coefficient of variation of service time cv = σ/E[S].
    c_a2 : float
        Squared coefficient of variation of interarrival times c_a²= σ_a²/E[A]².
        c_a2=1 → Poisson (M/G/c), c_a2<1 → regular arrivals,
        c_a2>1 → bursty arrivals.

    Returns
    -------
    float
        Mean waiting time Wq in hours. Returns 0.0 if λ=0, math.inf if ρ≥1.

    References
    ----------
    Allen, A.O., Cunneen, J.C. (1964). Queueing models for computer
        communications system analysis. Proceedings of the IEEE, 52(12).
    Whitt, W. (1993). Approximations for the GI/G/m queue.
        Production and Operations Management, 2(2), 114–161.
    Adan, I., Resing, J. (2015). Queueing Theory. TU Eindhoven. pp. 57–59.
    """

    if lambda_ == 0.0:
        return 0.0
    rho = lambda_ / (c * mu)
    if rho >= 1.0:
        return math.inf
    ec = _ref_erlang_c(c, rho)
    return (ec / (c * mu - lambda_)) * (c_a2 + cv**2) / 2


def ref_wq(
    lambda_: float, mu: float, c: int, cv: float, method: str, c_a2: float = 1.0
) -> float:
    """Dispatcher to get the fitting reference formula."""
    if method == "lee_longton":
        return ref_wq_lee_longton(lambda_, mu, c, cv)
    elif method == "allen_cunneen":
        return ref_wq_allen_cunneen(lambda_, mu, c, cv, c_a2)
    raise ValueError(f"Unbekannte Methode: {method}")


def stability_lower_bound(lambda_: float, mu: float) -> int:
    return math.ceil(lambda_ / mu)


@pytest.fixture(scope="session")
def standard_params() -> Dict[str, Any]:
    """standard parameters for Lee-Longton."""
    return {
        "lambda_target": 10.0,
        "charging_time_min": 45,
        "stdev_ct_min": 10,
        "waiting_times_min": [5, 10, 20, 60],
        "method": "lee_longton",
        "beta": 1.0,
        "max_server": 20,
    }


@pytest.fixture(scope="session")
def allen_cunneen_params() -> Dict[str, Any]:
    """standard parameter for Allen-Cunneen (c_a2 > 1 → higher variance of interarrival times)."""
    return {
        "lambda_target": 10.0,
        "charging_time_min": 45,
        "stdev_ct_min": 10,
        "waiting_times_min": [5, 10, 20, 60],
        "method": "allen_cunneen",
        "c_a2": 1.5,
        "beta": 1.0,
        "max_server": 25,
    }
