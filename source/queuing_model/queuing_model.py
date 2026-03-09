import pandas as pd
import math
import warnings
from typing import Tuple, List, Literal, Annotated
from pydantic import Field, validate_call
from functools import wraps

_FACTORS = {"hours_to_minutes": 60, "hours_to_seconds": 3600, "hours_to_days": 1 / 24}
_MINUTES_TO_HOURS = 1 / 60
_PER_MINUTE_TO_HOURS = 60
_DEFAULT_TIME_COLS = ["wq_mmc", "wq_mgc"]

_WQ_COL = {
    "lee_longton": "wq_mgc",
    "lee_longton_old": "wq_mgc",
    "allen_cunneen": "wq_gigc",
}


def convert_units(
    time_map: dict = None,
    rate_map: dict = None,
):
    """Convert units for function inputs and outputs.

    Parameters
    ----------
    time_map : dict, optional
        ``{minutes_kwarg: (hours_kwarg, target_kwarg)}`` — if the caller
        provides ``minutes_kwarg``, it is converted to hours and passed as
        ``target_kwarg``. If ``hours_kwarg`` is provided instead, it is
        passed as ``target_kwarg`` unchanged. List values are supported,
        factor is applied element-wise.
    rate_map : dict, optional
        ``{per_min_kwarg: (per_hour_kwarg, target_kwarg)}`` — if the caller
        provides ``per_min_kwarg``, it is converted to per hour (*60) and
        passed as ``target_kwarg``. If ``per_hour_kwarg`` is provided instead,
        it is passed as ``target_kwarg`` unchanged.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Duration conversion: minutes → hours
            if time_map:
                for min_key, (hours_key, target_key) in time_map.items():
                    if min_key in kwargs:
                        v = kwargs.pop(min_key)
                        kwargs[target_key] = (
                            [x * _MINUTES_TO_HOURS for x in v]
                            if isinstance(v, list)
                            else v * _MINUTES_TO_HOURS
                        )
                    elif hours_key in kwargs:
                        kwargs[target_key] = kwargs.pop(hours_key)

            # Rate conversion: per minute → per hour
            if rate_map:
                for min_key, (hours_key, target_key) in rate_map.items():
                    if min_key in kwargs:
                        v = kwargs.pop(min_key)
                        kwargs[target_key] = v * _PER_MINUTE_TO_HOURS
                    elif hours_key in kwargs:
                        kwargs[target_key] = kwargs.pop(hours_key)

            result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator


def _get_factor(direction: str) -> float:
    """Look up the unit conversion factor for a given direction.

    Parameters
    ----------
    direction : str
        Conversion key. Must be one of ``'hours_to_minutes'``,
        ``'hours_to_seconds'``, or ``'hours_to_days'``.

    Returns
    -------
    float
        Multiplication factor to convert a value expressed in hours to the
        target unit.

    Raises
    ------
    ValueError
        If ``direction`` is not a recognised key in ``_FACTORS``.
    """
    if direction not in _FACTORS:
        raise ValueError(
            f"Unknown direction '{direction}'. "
            f"Valid options: {set(_FACTORS.keys())}"
        )
    return _FACTORS[direction]


def _convert_units_dataframe(result: pd.DataFrame, output: str | dict) -> pd.DataFrame:
    """Apply unit conversion to selected columns of a DataFrame.

    Parameters
    ----------
    result : pd.DataFrame
        DataFrame whose columns are to be converted. Modified in-place.
    output : str or dict
        If a ``str``, the corresponding factor from ``_FACTORS`` is applied to
        all float-typed columns. If a ``dict`` of the form
        ``{column: direction}``, each named column is scaled by its individual
        factor.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with converted column values.
    """
    if isinstance(output, dict):
        for col, direction in output.items():
            if col in result.columns:
                result[col] = result[col] * _get_factor(direction)
    else:
        float_cols = result.select_dtypes(include="float").columns
        result[float_cols] = result[float_cols] * _get_factor(output)

    return result


def server_utilization(lambda_target: float, servers: int, mu: float) -> float:
    """Compute the average server utilization ρ = λ / (c·μ).

    Parameters
    ----------
    lambda_target : float
        Arrival rate λ in units per hour.
    servers : int
        Number of parallel servers c.
    mu : float
        Service rate per server μ in units per hour.

    Returns
    -------
    float
        Server utilization ρ.
    """
    if servers * mu <= 0:
        raise ValueError("Invalid parameters")
    return lambda_target / (servers * mu)


@validate_call
def queue_mgc_lee_longton_old(
    mean_waiting_time: Annotated[float, Field(ge=0)],
    server: Annotated[int, Field(gt=0)],
    mu: Annotated[float, Field(gt=0)],
    charging_time: Annotated[float, Field(ge=0)],
    cv: Annotated[float, Field(ge=0)],
    wq_mgc_init: Annotated[float, Field(gt=1)] = 50,
    roh_start: Annotated[float, Field(gt=0, lt=1)] = 0.99,
):
    """M/G/c waiting-time approximation using the Lee–Longton (1959) scaling factor.

    Computes the maximum feasible arrival rate λ for a multi-server queue by
    iteratively reducing λ until the mean waiting time no longer exceeds the
    target.

    The exact M/M/c waiting time (Erlang-C) is scaled by the factor
    (1 + cv²) / 2, taken from the Pollaczek–Khinchine formula, to account for
    the variability of service times in an M/G/c system. This scaling factor
    originates from Lee and Longton (1959) and has been applied in related
    contexts by Funke (2018).

    .. warning::
        This implementation contains a known bug: ``wq_part2`` incorrectly
        includes the last term in the summation, and the range is one step too
        short. The function is retained for backward compatibility and
        comparability purposes only. Use :func:`queue_mgc_lee_longton` instead.

    Parameters
    ----------
    mean_waiting_time : float
        Target mean waiting time in hours.
    server : int
        Number of parallel servers (charging points).
    mu : float
        Mean service rate per server in 1/hours.
    charging_time : float
        Average service (charging) time per customer in hours.
    cv : float
        Coefficient of variation of service times (standard deviation / mean).
    wq_mgc_init : float, optional
        Initial waiting time guess in hours. Default is 50.
    roh_start : float, optional
        Initial utilization factor ρ₀. Default is 0.99.

    Returns
    -------
    list[float]
        ``[λmax (1/h), ρ, Wq_MM_c (hours), Wq_MG_c (hours), Wq/ServiceTime ratio]``
    """
    wq_mgc = wq_mgc_init
    lambda_max = roh_start * (server * mu)

    while wq_mgc > mean_waiting_time:
        lambda_max -= 0.0001

        roh = lambda_max / (server * mu)

        if server > 1:
            wq_part1 = (
                (1 / (1 - roh))
                * (1 / (server * mu))
                * (((server * roh) ** server) / math.factorial(server))
            )
            wq_part2 = (1 - roh) * sum(
                [
                    (
                        ((server * roh) ** n) / math.factorial(n)
                        + ((server * roh) ** server) / math.factorial(server)
                    )
                    for n in range(0, server - 1)
                ]
            )
            wq_mmc = wq_part1 * wq_part2**-1

        else:
            wq_mmc = (roh / (1 - roh)) * (charging_time / server)

        wq_mgc = wq_mmc * ((1 + cv**2) / 2)
        wz_az = wq_mgc / charging_time

    return [lambda_max, roh, wq_mmc, wq_mgc, wz_az]


def _logsumexp(values: List[float]) -> float:
    """Compute log(sum(exp(values))) in a numerically stable way.

    Parameters
    ----------
    values : list[float]
        Input values.

    Returns
    -------
    float
        log-sum-exp result.
    """
    m = max(values)
    if not math.isfinite(m):
        return float("inf")
    return m + math.log(sum(math.exp(v - m) for v in values))


def _erlang_c_prob_wait(c: int, rho: float) -> float:
    """Compute the Erlang-C waiting probability P_wait for an M/M/c system.

    Uses log-space arithmetic and ``lgamma`` for numerical stability.

    Parameters
    ----------
    c : int
        Number of servers.
    rho : float
        Server utilization ρ = λ / (c·μ). Must satisfy 0 ≤ ρ < 1.

    Returns
    -------
    float
        Probability that an arriving customer has to wait, P_wait ∈ [0, 1].
    """
    if c == 1:
        return rho  # M/M/1 special case: P_wait = rho

    if rho <= 0:
        return 0.0
    if rho >= 1:
        return 1.0

    cr = c * rho
    log_cr = math.log(cr)

    # log(a_n) = n*log(cr) - log(n!) for n = 0..c-1
    log_a = [(n * log_cr) - math.lgamma(n + 1) for n in range(0, c)]
    log_sum_terms = _logsumexp(log_a)

    # log(a_c)
    log_a_c = c * log_cr - math.lgamma(c + 1)

    # ratio = ((1-rho)*sum_terms) / a_c  in log-space
    log_ratio = math.log1p(-rho) + log_sum_terms - log_a_c

    # Guard against extreme values
    if log_ratio > 700:  # exp(700) ~ 1e304
        return 0.0
    if log_ratio < -700:  # exp(-700) ~ 0
        return 1.0

    ratio = math.exp(log_ratio)
    return 1.0 / (1.0 + ratio)


def _compute_wq_for_lambda(
    lmbda: float,
    c: int,
    mu: float,
    charging_time: float,
    cv: float,
    c_a2: float,
) -> Tuple[float, float, float]:
    """Compute ρ, Wq_MM_c, and Wq_GI_G_c or Wq_MG_c (c_a² = 1.0) for a given arrival rate λ.

    For c_a² > 1
    For c_a² = 1.0 (Poisson arrivals) this function computes the waiting times in a M/G/c system.

    Parameters
    ----------
    lmbda : float
        Arrival rate λ in 1/hours.
    c : int
        Number of servers.
    mu : float
        Service rate per server in 1/hours, i.e. ``1 / charging_time``.
    charging_time : float
        Mean service time in hours.
    cv : float
        Coefficient of variation of service times (standard deviation / mean).
    c_a2 : float
        Squared coefficient of variation of interarrival times.
        Use 1.0 for Poisson arrivals (reduces to Lee-Longton).

    Returns
    -------
    tuple[float, float, float]
        ``(ρ, Wq_MM_c [hours], Wq_GI_G_c [hours])``
    """
    if lmbda <= 0:
        return 0.0, 0.0, 0.0

    roh = lmbda / (c * mu)

    if roh >= 1.0:
        return roh, float("inf"), float("inf")

    # -----------------------------
    # GI/G/1 | M/G/1 (c_a² = 1.0) special case
    # -----------------------------
    if c == 1:
        E_S = charging_time
        E_S2 = E_S * E_S * (1 + cv * cv)
        # Pollaczek-Khinchine base, scaled by (c_a2 + c_s2) / 2
        # For c_a2=1.0 this reduces to the standard P-K formula
        wq_mg1 = (lmbda * E_S2) / (2 * (1 - roh))
        wq_gigc = wq_mg1 * ((c_a2 + cv * cv) / (1 + cv * cv))
        return roh, wq_mg1, wq_gigc

    # -----------------------------
    # M/M/c base (Erlang-C)
    # -----------------------------
    P_wait = _erlang_c_prob_wait(c, roh)
    denom = c * mu * (1 - roh)
    wq_mmc = P_wait / denom

    # -----------------------------
    # Allen-Cunneen GI/G/c approximation | M/G/c Lee-Longton approximation (c_a² = 1.0)
    # -----------------------------
    wq_gigc = wq_mmc * ((c_a2 + cv * cv) / 2.0)

    return roh, wq_mmc, wq_gigc


def _bisect_lambda(
    mean_waiting_time: float,
    server: int,
    mu: float,
    charging_time: float,
    cv: float,
    c_a2: float,
    roh_start: float,
    tol_minutes: float,
    max_iter: int,
) -> List[float]:
    """Bisection core shared by queue_mgc_lee_longton and queue_gigc_allen_cunneen.
        Determines the maximum arrival rate λ for a target mean waiting time.

    Parameters
    ----------
    mean_waiting_time : float
        Target mean waiting time in hours.
    server : int
        Number of parallel servers c.
    mu : float
        Mean service rate per server in 1/hours.
    charging_time : float
        Mean service (charging) time per customer in hours.
    cv : float
        Coefficient of variation of service times.
    c_a2 : float
        Squared coefficient of variation of interarrival times.
        Use 1.0 for Lee-Longton (Poisson arrivals).
    roh_start : float
        Initial upper utilization bound ρ₀ (< 1).
    tol_minutes : float
        Convergence tolerance on the waiting time.
    max_iter : int
        Maximum number of bisection iterations.

    Returns
    -------
    list[float]
        ``[λmax (1/h), ρ, Wq_MMc (hours), Wq_Model (hours), Wq/ServiceTime ratio]``
    """
    lambda_max = server * mu * roh_start
    lambda_low = 0.0

    best_lambda = 0.0
    best_roh = 0.0
    best_wq = 0.0
    best_wq_out = 0.0

    for _ in range(max_iter):
        mid = 0.5 * (lambda_low + lambda_max)

        roh, wq_mmc, wq_out = _compute_wq_for_lambda(
            mid, server, mu, charging_time, cv, c_a2
        )

        if not math.isfinite(wq_out):
            lambda_max = mid
            continue

        if wq_out <= mean_waiting_time:
            best_lambda = mid
            best_roh = roh
            best_wq = wq_mmc
            best_wq_out = wq_out
            lambda_low = mid
        else:
            lambda_max = mid

        if abs(wq_out - mean_waiting_time) < tol_minutes:
            break
        if lambda_max - lambda_low < 1e-12:
            break

    if best_lambda == 0.0 and mean_waiting_time > 0:
        raise ValueError(
            "No lambda found that meets the waiting-time target; try higher server count."
        )

    return [best_lambda, best_roh, best_wq, best_wq_out, best_wq_out / charging_time]


@validate_call
def queue_mgc_lee_longton(
    mean_waiting_time: Annotated[float, Field(ge=0)],
    server: Annotated[int, Field(gt=0)],
    mu: Annotated[float, Field(gt=0)],
    charging_time: Annotated[float, Field(ge=0)],
    cv: Annotated[float, Field(ge=0)],
    roh_start: Annotated[float, Field(gt=0, lt=1)] = 0.999999,
    tol_minutes: Annotated[float, Field(gt=0)] = 1e-5,
    max_iter: Annotated[int, Field(gt=10, lt=1000000)] = 80,
) -> List[float]:
    """Numerically stable M/G/c approximation using the Lee–Longton (1959) scaling factor.

    Determines the maximum arrival rate λ for a target mean waiting time using
    a bisection approach. Scales the exact M/M/c (Erlang-C) waiting time by
    the Lee–Longton factor (1 + cv²) / 2 to approximate the M/G/c waiting time.

    Delegates to :func:`_bisect_lambda` with c_a² = 1.0 (Poisson arrivals).

    Parameters
    ----------
    mean_waiting_time : float
        Target mean waiting time in hours.
    server : int
        Number of parallel servers c.
    mu : float
        Mean service rate per server in 1/hours.
    charging_time : float
        Mean service (charging) time per customer in hours.
    cv : float
        Coefficient of variation of service times (standard deviation / mean).
    roh_start : float, optional
        Initial upper utilization bound ρ₀ (< 1). Default is 0.999999.
    tol_minutes : float, optional
        Convergence tolerance on the waiting time. Default is 1e-5.
    max_iter : int, optional
        Maximum number of bisection iterations. Default is 80.

    Returns
    -------
    list[float]
        ``[λmax (1/h), ρ, Wq_MMc (hours), Wq_MGc (hours), Wq/ServiceTime ratio]``

    References
    ----------
    Lee, A. M., Longton, P. A. (1959). Queueing processes associated
        with airline passenger check-in. Operational Research Quarterly,
        10, 56–71.
    Pollaczek, F. (1930). Über eine Aufgabe der Wahrscheinlichkeitstheorie.
        Mathematische Zeitschrift, 32, 64–100.
    Khinchin, A. Y. (1932). Mathematical theory of a stationary queue.
        Matematicheskii Sbornik, 39(4), 73–84.
    """
    return _bisect_lambda(
        mean_waiting_time,
        server,
        mu,
        charging_time,
        cv,
        c_a2=1.0,
        roh_start=roh_start,
        tol_minutes=tol_minutes,
        max_iter=max_iter,
    )


@validate_call
def queue_gigc_allen_cunneen(
    mean_waiting_time: Annotated[float, Field(ge=0)],
    server: Annotated[int, Field(gt=0)],
    mu: Annotated[float, Field(gt=0)],
    charging_time: Annotated[float, Field(ge=0)],
    cv: Annotated[float, Field(ge=0)],
    c_a2: Annotated[float, Field(ge=0)],
    roh_start: Annotated[float, Field(gt=0, lt=1)] = 0.999999,
    tol_minutes: Annotated[float, Field(gt=0)] = 1e-5,
    max_iter: Annotated[int, Field(gt=10, lt=1000000)] = 80,
) -> List[float]:
    """GI/G/c waiting-time approximation using the Allen-Cunneen formula.

    Determines the maximum arrival rate λ for a target mean waiting time
    using a bisection approach. Extends Lee-Longton to non-Poisson arrivals
    by incorporating c_a² into the scaling factor (c_a² + cv²) / 2.

    For Poisson arrivals (c_a² = 1.0) equivalent to
    :func:`queue_mgc_lee_longton`. Delegates to :func:`_bisect_lambda`.

    Parameters
    ----------
    mean_waiting_time : float
        Target mean waiting time in hours.
    server : int
        Number of parallel servers c.
    mu : float
        Mean service rate per server in 1/hours.
    charging_time : float
        Mean service (charging) time per customer in hours.
    cv : float
        Coefficient of variation of service times (standard deviation / mean).
    c_a2 : float
        Squared coefficient of variation of interarrival times.
    roh_start : float, optional
        Initial upper utilization bound ρ₀ (< 1). Default is 0.999999.
    tol_minutes : float, optional
        Convergence tolerance on the waiting time. Default is 1e-5.
    max_iter : int, optional
        Maximum number of bisection iterations. Default is 80.

    Returns
    -------
    list[float]
        ``[λmax (1/h), ρ, Wq_MMc (hours), Wq_GIGc (hours), Wq/ServiceTime ratio]``

    References
    ----------
    Allen, A. O. (1978). Probability, Statistics, and Queueing Theory.
        Academic Press.
    Cooper, R. B. (1990). Introduction to Queueing Theory (3rd ed.),
        p. 508, Formula 9.3.
    Lee, A. M., Longton, P. A. (1959). Queueing processes associated
        with airline passenger check-in. Operational Research Quarterly,
        10, 56–71.
    """
    return _bisect_lambda(
        mean_waiting_time,
        server,
        mu,
        charging_time,
        cv,
        c_a2=c_a2,
        roh_start=roh_start,
        tol_minutes=tol_minutes,
        max_iter=max_iter,
    )


@convert_units(
    time_map={
        "charging_time_min": ("charging_time_hours", "charging_time"),
        "stdev_ct_min": ("stdev_ct_hours", "stdev_ct"),
        "mean_waiting_time_min": ("mean_waiting_time_hours", "mean_waiting_time"),
    }
)
@validate_call
def que_mgc(
    charging_time: Annotated[float, Field(ge=0)],
    stdev_ct: Annotated[float, Field(ge=0)],
    mean_waiting_time: Annotated[float, Field(gt=0)],
    max_server: Annotated[int, Field(gt=0)],
    method: Literal["allen_culleen", "lee_longton", "lee_longton_old"],
    c_a2: Annotated[float | None, Field(ge=0)] = None,
    output_unit: (
        Literal["hours_to_minutes", "hours_to_seconds", "hours_to_days"] | None
    ) = "hours_to_minutes",
    output_cols: list[str] | None = Field(
        default_factory=lambda: list(_DEFAULT_TIME_COLS)
    ),  # noqa
) -> pd.DataFrame:
    """Compute the maximum arrival rate across server counts for an M/G/c queue.

    Iterates over server counts from 1 to ``max_server`` and computes the
    maximum feasible arrival rate λ, traffic intensity ρ, mean waiting times,
    and the waiting-to-service-time ratio for each count.

    Parameters
    ----------
    charging_time : float
        Mean service (charging) time in hours.
    stdev_ct : float
        Standard deviation of the service time in hours.
    mean_waiting_time : float
        Target mean waiting time in hours.
    max_server : int
        Maximum number of servers to evaluate.
    c_a2 : float | None
        Squared coefficient of variation of interarrival times. Only necessary for 'alleen-culleen'
    method : {'allen_culleen', 'lee_longton', 'lee_longton_old'}
        Queueing approximation method to use.
    output_unit : {'hours_to_minutes', 'hours_to_seconds', 'hours_to_days'} or None, optional
        Time unit for output columns. ``None`` keeps hours. Default is
        ``'hours_to_minutes'``.
    output_cols : list[str] or None, optional
        Columns to apply the unit conversion to. Default is
        ``['wq_mmc', 'wq_mgc']``.

    Unit Handling
    -------------
    Each time parameter can be provided in either minutes or hours.
    Exactly one variant must be given per parameter:

    +-------------------------+-------------------------+
    | Minutes                 | Hours                   |
    +=========================+=========================+
    | charging_time_min       | charging_time_hours     |
    +-------------------------+-------------------------+
    | stdev_ct_min            | stdev_ct_hours          |
    +-------------------------+-------------------------+
    | mean_waiting_time_min   | mean_waiting_time_hours |
    +-------------------------+-------------------------+

    If the parameter is not specified the default of the function is used.

    Returns
    -------
    pd.DataFrame
        One row per server count with columns
        ``['servers', 'lambda', 'roh', 'wq_mmc', 'wq_mgc', 'wz/az']``.
    """
    dict_method = {
        "allen_cunneen": queue_gigc_allen_cunneen,
        "lee_longton_old": queue_mgc_lee_longton_old,
        "lee_longton": queue_mgc_lee_longton,
    }

    # Validation
    if method == "allen_cunneen" and c_a2 is None:
        raise ValueError("c_a2 is required for method='allen_cunneen'.")
    if method != "allen_cunneen" and c_a2 is not None:
        warnings.warn("c_a2 is ignored for method != 'allen_cunneen'.", UserWarning)

    mu = 1 / charging_time
    cv = stdev_ct / charging_time

    wq_model_h = _WQ_COL[method]
    cols = ["servers", "lambda", "roh", "wq_mmc", wq_model_h, "wz/az"]

    queue = pd.DataFrame(
        0.0,
        index=list(range(1, max_server + 1)),
        columns=cols,
    )

    for server in range(1, max_server + 1):
        if method == "allen_cunneen":
            result = queue_gigc_allen_cunneen(
                mean_waiting_time, server, mu, charging_time, cv, c_a2
            )
        else:
            result = dict_method[method](
                mean_waiting_time, server, mu, charging_time, cv
            )
        queue.loc[server, cols] = [server] + result

    if output_cols is None:
        output_cols = ["wq_mmc", wq_model_h]

    if output_unit is not None:
        queue = _convert_units_dataframe(
            queue,
            output={col: output_unit for col in output_cols},
        )

    return queue


@convert_units(
    time_map={
        "charging_time_min": ("charging_time_hours", "charging_time"),
        "stdev_ct_min": ("stdev_ct_hours", "stdev_ct"),
        "waiting_times_min": ("waiting_times_hours", "waiting_times"),
    },
    rate_map={"lambda_target_min": ("lambda_target_hours", "lambda_target")},
)
@validate_call
def que_mgc_server_wq(
    lambda_target: Annotated[float, Field(gt=0)],
    charging_time: Annotated[float, Field(ge=0)],
    stdev_ct: Annotated[float, Field(ge=0)],
    waiting_times: list[Annotated[float, Field(gt=0)]],
    method: Literal["allen_culleen", "lee_longton", "lee_longton_old"],
    c_a2: Annotated[float | None, Field(ge=0)] = None,
    max_server: Annotated[int, Field(gt=0)] = 1000,
    output_unit: (
        Literal["hours_to_minutes", "hours_to_seconds", "hours_to_days"] | None
    ) = "hours_to_minutes",
) -> tuple:
    """Determine the minimum number of servers required to handle a target arrival rate
    under various mean waiting-time constraints.

    For each waiting-time target in ``waiting_times``, finds the smallest server
    count c such that the system can stably process ``lambda_target`` arrivals per
    hour while keeping the mean waiting time at or below the specified target.
    The search increments c from 1 upward and stops as soon as the maximum
    feasible arrival rate λ_max(c) first exceeds ``lambda_target``.

    Parameters
    ----------
    lambda_target : float
        Target arrival rate λ in units per hour.
    charging_time : float
        Mean service (charging) time in hours.
    stdev_ct : float
        Standard deviation of the service time in hours.
    waiting_times : list[float]
        Target mean waiting times in hours to evaluate.
    c_a2 : float | None
        Squared coefficient of variation of interarrival times. Only necessary for 'alleen-culleen'
    method : {'coop', 'lee_longton', 'lee_longton_old'}
        Queueing approximation method to use.
    max_server : int, optional
        Maximum number of servers to consider. Default is 1000.
    output_unit : {'hours_to_minutes', 'hours_to_seconds', 'hours_to_days'} or None, optional
        Time unit for the waiting-time keys in the returned dictionary.
        ``None`` keeps hours. Default is ``'hours_to_minutes'``.

    Unit Handling
    -------------
    Each time parameter can be provided in either minutes or hours:

    +----------------------+---------------------+
    | Minutes              | Hours               |
    +======================+=====================+
    | charging_time_min    | charging_time_hours |
    +----------------------+---------------------+
    | stdev_ct_min         | stdev_ct_hours      |
    +----------------------+---------------------+
    | waiting_times_min    | waiting_times_hours |
    +----------------------+---------------------+
    | lambda_target_min    | lambda_target_hours |
    +----------------------+---------------------+

    If the parameter is not specified the default of the function is used.

    Returns
    -------
    tuple
        ``(lambda_target, dict_server_wq)`` where ``dict_server_wq`` maps
        each mean waiting time (in ``output_unit``) to the minimum required
        number of servers.
    """
    dict_method = {
        "allen_cunneen": queue_gigc_allen_cunneen,
        "lee_longton_old": queue_mgc_lee_longton_old,
        "lee_longton": queue_mgc_lee_longton,
    }

    # Validation
    if method == "allen_cunneen" and c_a2 is None:
        raise ValueError("c_a2 is required for method='allen_cunneen'.")
    if method != "allen_cunneen" and c_a2 is not None:
        warnings.warn("c_a2 is ignored for method != 'allen_cunneen'.", UserWarning)

    dict_server_wq = {}

    mu = 1 / charging_time
    cv = stdev_ct / charging_time

    for mean_waiting_time in waiting_times:
        chosen_server = None

        for server in range(1, max_server + 1):
            if method == "allen_cunneen":
                lambda_max, roh, wq_mmc_h, wq_model_h, wz_az = queue_gigc_allen_cunneen(
                    mean_waiting_time, server, mu, charging_time, cv, c_a2
                )
            else:
                lambda_max, roh, wq_mmc_h, wq_model_h, wz_az = dict_method[method](
                    mean_waiting_time, server, mu, charging_time, cv
                )

            if mean_waiting_time > 0 and lambda_max <= 0:
                continue

            if lambda_max > lambda_target:
                chosen_server = server
                break

        # Unit conversion
        if output_unit is not None:
            factor = _FACTORS[output_unit]
            mean_waiting_time = mean_waiting_time * factor

        dict_server_wq[str(mean_waiting_time)] = chosen_server

    return lambda_target, dict_server_wq


@convert_units(
    time_map={
        "charging_time_min": ("charging_time_hours", "charging_time"),
        "stdev_ct_min": ("stdev_ct_hours", "stdev_ct"),
    }
)
def _qed_servers(lambda_rate, mu, beta=1.0):
    """Compute the initial server count using the QED (Halfin–Whitt) staffing rule.

    Applies the square-root staffing formula:
    c ≈ R + β·√R, where R = λ / μ is the offered load.

    Parameters
    ----------
    lambda_rate : float
        Arrival rate λ in 1/hours.
    mu : float
        Service rate per server μ in 1/hours.
    beta : float, optional
        QED quality parameter. Default is 1.0.

    Returns
    -------
    int
        Ceiling of the staffing estimate.
    """
    R = lambda_rate / mu
    return math.ceil(R + beta * math.sqrt(R))


def _auto_search_radius(lambda_target, mu):
    """Derive an automatic server search radius from the offered load.

    Computes a heuristic search window around the QED initial server estimate
    based on the square root of the offered load R = λ / μ. The radius grows
    with R to ensure the local search remains comprehensive as load increases.

    Parameters
    ----------
    lambda_target : float
        Target arrival rate λ in 1/hours.
    mu : float
        Service rate per server μ in 1/hours.

    Returns
    -------
    int
        Search radius, at least 5 and otherwise ``round(2 · √R)``.
    """
    R = lambda_target / mu
    search_radius = int(round(max(5, 2 * math.sqrt(R)), 0))
    return search_radius


@convert_units(
    time_map={
        "charging_time_min": ("charging_time_hours", "charging_time"),
        "stdev_ct_min": ("stdev_ct_hours", "stdev_ct"),
        "waiting_times_min": ("waiting_times_hours", "waiting_times"),
    },
    rate_map={"lambda_target_min": ("lambda_target_hours", "lambda_target")},
)
@validate_call
def que_mgc_server_wq_qed(
    lambda_target: Annotated[float, Field(gt=0)],
    charging_time: Annotated[float, Field(gt=0)],
    stdev_ct: Annotated[float, Field(ge=0)],
    waiting_times: list[Annotated[float, Field(ge=0)]],
    method: Literal["allen_culleen", "lee_longton", "lee_longton_old"],
    c_a2: Annotated[float | None, Field(ge=0)] = None,
    beta: Annotated[float, Field(ge=0)] = 1.0,
    search_radius: Annotated[(int | None), Field(gt=1)] = None,
    max_server: Annotated[int, Field(gt=0)] = 1000,
    output_unit: (
        Literal["hours_to_minutes", "hours_to_seconds", "hours_to_days"] | None
    ) = "hours_to_minutes",
) -> tuple:
    """Determine required server counts via QED-guided search for an M/G/c queue.

    For each target mean waiting time, finds the smallest number of servers that
    can stably serve ``lambda_target`` and satisfies the waiting-time constraint.
    The search is initialized using the QED (Halfin–Whitt) square-root staffing
    rule c ≈ R + β·√R, with R = λ / μ.

    Parameters
    ----------
    lambda_target : float
        Target arrival rate λ in units per hour.
    charging_time : float
        Mean service (charging) time in hours.
    stdev_ct : float
        Standard deviation of the service (charging) time in hours.
    waiting_times : list[float]
        Target mean waiting times in hours for which the required server count
        is to be determined.
    c_a2 : float | None
        Squared coefficient of variation of interarrival times. Only necessary for 'alleen-culleen'
    method : {'coop', 'lee_longton', 'lee_longton_old'}
        Queueing approximation used to evaluate mean waiting times.
        ``'lee_longton_old'`` retains a known summation bug for comparability.
    beta : float, optional
        QED quality parameter. ``beta = 0`` corresponds to efficiency-driven
        staffing; higher values add safety capacity. Default is 1.0.
    search_radius : int or None, optional
        Search window around the QED initial estimate. Used for computational
        efficiency only and has no queueing-theoretic interpretation. If
        ``None``, an automatic radius is derived from the offered load.
    max_server : int, optional
        Maximum number of servers considered in the search. Default is 1000.
    output_unit : {'hours_to_minutes', 'hours_to_seconds', 'hours_to_days'} or None, optional
        Time unit for the waiting-time keys in the returned dictionary.
        ``None`` keeps hours. Default is ``'hours_to_minutes'``.

    Unit Handling
    -------------
    Each time parameter can be provided in either minutes or hours:

    +----------------------+---------------------+
    | Minutes              | Hours               |
    +======================+=====================+
    | charging_time_min    | charging_time_hours |
    +----------------------+---------------------+
    | stdev_ct_min         | stdev_ct_hours      |
    +----------------------+---------------------+
    | waiting_times_min    | waiting_times_hours |
    +----------------------+---------------------+
    | lambda_target_min    | lambda_target_hours |
    +----------------------+---------------------+

    If the parameter is not specified the default of the function is used.

    Returns
    -------
    tuple
        ``(lambda_target, dict_server_wq)`` where ``dict_server_wq`` maps each
        target mean waiting time (in ``output_unit``) to the minimum feasible
        server count, or ``None`` if no solution was found within the search
        range.
    """
    dict_method = {
        "allen_cunneen": queue_gigc_allen_cunneen,
        "lee_longton_old": queue_mgc_lee_longton_old,
        "lee_longton": queue_mgc_lee_longton,
    }

    # Validation
    if method == "allen_cunneen" and c_a2 is None:
        raise ValueError("c_a2 is required for method='allen_cunneen'.")
    if method != "allen_cunneen" and c_a2 is not None:
        warnings.warn("c_a2 is ignored for method != 'allen_cunneen'.", UserWarning)

    dict_server_wq = {}

    stdev_ct_h = stdev_ct / 60
    mu = 1 / charging_time
    cv = stdev_ct_h / charging_time

    # Minimum server count for stability
    min_stable_servers = math.ceil(lambda_target / mu)

    if search_radius is None:
        search_radius = _auto_search_radius(lambda_target, mu)
        print(f"Auto search_radius = {search_radius:.1f}")

    for mean_waiting_time in waiting_times:

        # --- QED initial guess ---
        c_qed = _qed_servers(lambda_target, mu, beta)

        search_start = max(min_stable_servers, c_qed - search_radius)
        search_end = min(max_server, c_qed + search_radius)

        # --- Local search around QED (full search, no early break) ---
        feasible_servers = []

        for server in range(search_start, search_end + 1):

            if method == "allen_cunneen":
                lambda_max, roh, wq_mmc_h, wq_model_h, wz_az = queue_gigc_allen_cunneen(
                    mean_waiting_time, server, mu, charging_time, cv, c_a2
                )
            else:
                lambda_max, roh, wq_mmc_h, wq_model_h, wz_az = dict_method[method](
                    mean_waiting_time, server, mu, charging_time, cv
                )

            # Server is feasible if it can serve lambda_target and meets waiting time
            if lambda_max >= lambda_target:
                feasible_servers.append(server)

        # Pick minimal feasible server to honor QED
        if feasible_servers:
            best_c = min(feasible_servers)
        else:
            best_c = None
            print(
                f"Warning: No server satisfies target Wq={mean_waiting_time} h at λ={lambda_target} h⁻¹"
            )

        # Unit conversion
        if output_unit is not None:
            factor = _FACTORS[output_unit]
            mean_waiting_time = mean_waiting_time * factor
        dict_server_wq[str(mean_waiting_time)] = best_c

    return lambda_target, dict_server_wq
