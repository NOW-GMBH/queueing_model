import pandas as pd
import math
from typing import Tuple, List, Literal, Annotated
from pydantic import Field, validate_call


def server_utilization(lambda_target: float, servers: int, mu: float) -> float:
    """average server utilization ρ = λ/(cμ)"""
    if servers * mu <= 0:
        raise ValueError("Invalid parameters")
    return lambda_target / (servers * mu)


@validate_call
def queue_mgc_coop(
    mean_waiting_time: Annotated[float, Field(ge=0)],
    server: Annotated[int, Field(gt=0)],
    mu: Annotated[float, Field(gt=0)],
    charging_time: Annotated[float, Field(gt=0)],
    cv: Annotated[float, Field(ge=0)],
    wq_mgc_init: Annotated[float, Field(gt=1)] = 50,
    roh_start: Annotated[float, Field(gt=0, lt=1)] = 0.99,
    max_iterations: Annotated[int, Field(gt=10, lt=1000000)] = 1000000,
) -> List[float]:
    """
    Approximate M/G/c queueing model based on Cooper (1990, p.508, Eq. 9.3).

    Iteratively determines the maximum arrival rate (λ₀) for a multi-server queue
    such that the mean waiting time does not exceed a target value.
    The method uses Cooper’s simplified approximation for M/M/1 systems and
    extends it to M/G/c by applying a correction factor based on the coefficient
    of variation of service times (vₖ).

    :param mean_waiting_time: Target mean waiting time in hours.
    :param server: Number of parallel servers (charging points).
    :param mu: Mean service rate per server [1/hours].
    :param charging_time: Average service (charging) time per customer in hours.
    :param cv: Coefficient of variation of service times (standard deviation / mean).
    :param wq_mgc_init: Initial waiting time guess (in hours). Default is 50.
    :param roh_start: Initial utilization factor (ρ₀), default is 0.99.
    :return:  list
        [λmax (1/h), ρ, Wq_MM_c (hours), Wq_MG_c (hours), Wq/ServiceTime ratio]
    """
    # After Cooper 1990 - S.508 - Formel 9.3

    lambda_max = roh_start * (server * mu)
    roh = roh_start
    wq_mmc = 0.0
    wq_mgc = wq_mgc_init
    wz_az = 0.0

    iterations = 0
    while wq_mgc > mean_waiting_time:

        iterations += 1
        if iterations >= max_iterations:
            raise ValueError(f"Did not converge in {max_iterations} iterations")

        lambda_max -= 0.0001
        if lambda_max <= 0:
            raise ValueError("Lambda became non-positive")

        roh = lambda_max / (server * mu)
        wq_mmc = (roh / (1 - roh)) * (charging_time / server)
        wq_mgc = wq_mmc * ((1 + cv**2) / 2)
        wz_az = wq_mgc / charging_time

    return [lambda_max, roh, wq_mmc, wq_mgc, wz_az]


@validate_call
def queue_mgc_Adan_Resing_old(
    mean_waiting_time: Annotated[float, Field(ge=0)],
    server: Annotated[int, Field(gt=0)],
    mu: Annotated[float, Field(gt=0)],
    charging_time: Annotated[float, Field(gt=0)],
    cv: Annotated[float, Field(ge=0)],
    wq_mgc_init: Annotated[float, Field(gt=1)] = 50,
    roh_start: Annotated[float, Field(gt=0, lt=1)] = 0.99,
):
    """
    Warteschlangenmodell im M/G/c-System nach Adan & Resing (2017)

    Approximierung nach Funke 2018 - https://urn.fi/urn:nbn:de:hebis:34-2018041155288 ⎘

    Jedoch falsche Implementierung wq_part2 ist die Summenberechung falsch, sie summiert fälschlicherweise auch den
    letzten Term mit und der Range ist um -1 zu kurz
    :param mean_waiting_time: Target mean waiting time in hours.
    :param server: Number of parallel servers (charging points).
    :param mu: Mean service rate per server [1/hours]
    :param charging_time: Average service (charging) time per customer in hours
    :param cv: Coefficient of variation of service times (standard deviation / mean)
    :param wq_mgc_init: Initial waiting time guess (in hours). Default is 50
    :param roh_start: Initial utilization factor (ρ₀), default is 0.99.
    :return: list
        [λmax (1/h), ρ, Wq_MM_c (hours), Wq_MG_c (hours), Wq/ServiceTime ratio]
    """
    # Adan_Resing
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
    """Numerisch stabile Berechnung von log(sum(exp(values)))."""
    m = max(values)
    if not math.isfinite(m):
        return float("inf")
    return m + math.log(sum(math.exp(v - m) for v in values))


def _erlang_c_prob_wait(c: int, rho: float) -> float:
    """
    Numerisch stabile Berechnung der Erlang-C Wartwahrscheinlichkeit P_wait
    für ein M/M/c-System.
    Nutzt Logspace + lgamma zur Stabilisierung.
    """
    if c == 1:
        return rho  # M/M/1-Spezialfall: P_wait = rho

    if rho <= 0:
        return 0.0
    if rho >= 1:
        return 1.0

    cr = c * rho
    log_cr = math.log(cr)

    # log(a_n) = n*log(cr) - log(n!) für n = 0..c-1
    log_a = [(n * log_cr) - math.lgamma(n + 1) for n in range(0, c)]
    log_sum_terms = _logsumexp(log_a)

    # log(a_c)
    log_a_c = c * log_cr - math.lgamma(c + 1)

    # ratio = ((1-rho)*sum_terms) / a_c  im Logspace
    log_ratio = math.log1p(-rho) + log_sum_terms - log_a_c

    # Stabilisierung gegen extreme Werte
    if log_ratio > 700:  # exp(700) ~ 1e304
        return 0.0
    if log_ratio < -700:  # exp(-700) ~ 0
        return 1.0

    ratio = math.exp(log_ratio)
    return 1.0 / (1.0 + ratio)


def _compute_wq_for_lambda(
    lmbda: float, c: int, mu: float, charging_time_hours: float, cv: float
) -> Tuple[float, float, float]:
    """
    Berechne roh, Wq_MM_c (in Hours) und Wq_MGc (in Hours)
    für eine gegebene Ankunftsrate λ.

    charging_time_hours: Bedienzeit *in Stunden*
    mu: Service rate pro Server = 1 / (charging_time_in_hours)
    """

    if lmbda <= 0:
        return 0.0, 0.0, 0.0

    roh = lmbda / (c * mu)

    if roh >= 1.0:
        return roh, float("inf"), float("inf")

    # -----------------------------
    # M/G/1 Fall
    # -----------------------------
    if c == 1:
        E_S = charging_time_hours
        E_S2 = E_S * E_S * (1 + cv * cv)
        wq_hours = (lmbda * E_S2) / (2 * (1 - roh))
        wq_mgc_hours = wq_hours
        return roh, wq_hours, wq_mgc_hours

    # -----------------------------
    # M/M/c Basis
    # -----------------------------
    P_wait = _erlang_c_prob_wait(c, roh)
    denom = c * mu * (1 - roh)
    wq_hours = P_wait / denom

    # -----------------------------
    # M/G/c (Funke) Approximation
    # -----------------------------
    wq_mgc_hours = wq_hours * ((1 + cv * cv) / 2.0)

    return roh, wq_hours, wq_mgc_hours


@validate_call
def queue_mgc_Adan_Resing_stable(
    mean_waiting_time: Annotated[float, Field(ge=0)],
    server: Annotated[int, Field(gt=0)],
    mu: Annotated[float, Field(gt=0)],
    charging_time: Annotated[float, Field(gt=0)],
    cv: Annotated[float, Field(ge=0)],
    roh_start: Annotated[float, Field(gt=0, lt=1)] = 0.999999,
    tol_minutes: Annotated[float, Field(gt=0)] = 1e-5,
    max_iter: Annotated[int, Field(gt=10, lt=1000000)] = 80,
) -> List[float]:
    """
    Stabil berechnete maximale Ankunftsrate lambda für eine Zielwartezeit
    (mean_waiting_time in Minuten).

    Parameter:
    - mean_waiting_time (hours)
    - server = c (Anzahl Server)
    - charging_time (hours)
    - cv = Variationskoeffizient

    Rückgabe: List
    [λmax (1/h), ρ, Wq_MM_c (hours), Wq_MG_c (hours), Wq/ServiceTime ratio]
    """

    # Obergrenze für λ: systemstabil knapp unter c*mu
    lambda_max = server * mu * roh_start
    lambda_low = 0.0

    best_lambda = 0.0
    best_roh = 0.0
    best_wq = 0.0
    best_wq_mgc = 0.0

    for _ in range(max_iter):
        mid = 0.5 * (lambda_low + lambda_max)

        roh, wq_mmc, wq_mgc = _compute_wq_for_lambda(mid, server, mu, charging_time, cv)

        if not math.isfinite(wq_mgc):
            lambda_max = mid
            continue

        if wq_mgc <= mean_waiting_time:
            best_lambda = mid
            best_roh = roh
            best_wq = wq_mmc
            best_wq_mgc = wq_mgc
            lambda_low = mid
        else:
            lambda_max = mid

        if abs(wq_mgc - mean_waiting_time) < tol_minutes:
            break
        if lambda_max - lambda_low < 1e-12:
            break

    # Verhältnis Warten/Bedienen
    wz_az = best_wq_mgc / charging_time

    if best_lambda == 0.0 and mean_waiting_time > 0:
        raise ValueError(
            "No lambda found that meets the waiting-time target; try higher server count."
        )

    return [best_lambda, best_roh, best_wq, best_wq_mgc, wz_az]


@validate_call
def que_mgc(
    charging_time_min: Annotated[int, Field(gt=0)],
    stdev_ct_min: Annotated[int, Field(ge=0)],
    mean_waiting_time_min: Annotated[float, Field(gt=0)],
    max_server: Annotated[int, Field(gt=0)],
    method: Literal["coop", "adan", "adan_old"],
) -> pd.DataFrame:
    """
    Calculates the maximum arrival rate for various server counts in a queueing model using different methods.

    This function iterates through a range of server counts and calculates the optimal arrival rate (lambda), traffic intensity (roh),
    average waiting time (wq), maximum queue length (wq_mgc), and other statistics based on the specified method.

    Parameters:
    - charging_time_min: Average service time in minutes
    - stdev_ct_min: Standard deviation of the service time in minutes
    - mean_waiting_time: Target average waiting time in minutes for the system
    - max_server: Maximum number of servers to consider (default is 1000)
    - method: Method to use for calculating maximum arrival rate ('coop', 'adan', or 'adan_old')

    Returns:
    - DataFrame containing the calculated parameters for each server count, including the number of servers,
      λmax (1/h), ρ, Wq_MM_c (hours), Wq_MG_c (hours), Wq/ServiceTime ratio
    """

    dict_method = {
        "coop": queue_mgc_coop,
        "adan_old": queue_mgc_Adan_Resing_old,
        "adan": queue_mgc_Adan_Resing_stable,
    }
    method = dict_method[method]

    charging_time_h = charging_time_min / 60
    stdev_ct_h = stdev_ct_min / 60
    mean_waiting_time_h = mean_waiting_time_min / 60
    mu = 1 / charging_time_h
    cv = stdev_ct_h / charging_time_h

    queue = pd.DataFrame(
        0.0,
        index=list(range(1, max_server + 1)),
        columns=["servers", "lambda", "roh", "wq_mmc", "wq_mgc", "wz/az"],
    )

    for server in range(1, max_server + 1):
        queue.loc[server, ["servers", "lambda", "roh", "wq_mmc", "wq_mgc", "wz/az"]] = [
            server
        ] + method(mean_waiting_time_h, server, mu, charging_time_h, cv)

    return queue


@validate_call
def que_mgc_server_wq(
    lambda_target: Annotated[float, Field(gt=0)],
    charging_time_min: Annotated[int, Field(gt=0)],
    stdev_ct_min: Annotated[int, Field(ge=0)],
    waiting_times_min: list[Annotated[int, Field(gt=0)]],
    method: Literal["coop", "adan", "adan_old"],
    max_server: Annotated[int, Field(gt=0)] = 1000,
) -> tuple:
    """
    Determines the number of servers required to meet a target arrival rate for various mean waiting times.

    This function iterates through different waiting times and calculates the optimal number of servers needed based on
    the specified method (e.g., Cooper or Adan-Resing). It returns a dictionary mapping each waiting time to the
    corresponding number of servers required to achieve an arrival rate less than or equal to the target.

    Parameters:
    - lambda_target: Target arrival rate (lambda) in units per hour
    - charging_time: Average service time in minutes
    - stdev_ct: Standard deviation of the service time in minutes
    - waiting_times: List of mean waiting times in minutes for which the number of servers is to be determined
    - method: Method to use for calculating optimal server count ('coop' or 'adan')
    - max_server: Maximum number of servers to consider (default is 1000)

    Returns:
    - Tuple containing the target arrival rate and a dictionary mapping each mean waiting time to the corresponding number of servers.
    """

    dict_method = {
        "coop": queue_mgc_coop,
        "adan_old": queue_mgc_Adan_Resing_old,
        "adan": queue_mgc_Adan_Resing_stable,
    }
    method = dict_method[method]

    dict_server_wq = {}

    charging_time_h = charging_time_min / 60
    stdev_ct_h = stdev_ct_min / 60
    mu = 1 / charging_time_h
    cv = stdev_ct_h / charging_time_h

    for mean_waiting_time_min in waiting_times_min:
        chosen_server = None
        mean_waiting_time_h = mean_waiting_time_min / 60

        for server in range(1, max_server + 1):

            lambda_max, roh, wq_mmc_h, wq_mgc_h, wz_az = method(
                mean_waiting_time_h, server, mu, charging_time_h, cv
            )
            if mean_waiting_time_min > 0 and lambda_max <= 0:
                continue

            if lambda_max > lambda_target:
                chosen_server = server
                break

        dict_server_wq[str(mean_waiting_time_min)] = chosen_server

    return lambda_target, dict_server_wq


@validate_call
def queue_wq_roh_coop(
    roh_range: Annotated[list[float], Field(ge=0)],
    server: Annotated[int, Field(gt=0)],
    charging_time_min: Annotated[int, Field(gt=0)],
    stdev_ct_min: Annotated[int, Field(ge=0)],
) -> pd.DataFrame:
    queue = pd.DataFrame(
        columns=["lambda", "server", "roh", "wq_mmc", "wq_mgc", "wz/az", "krit_wert"]
    )

    charging_time_h = charging_time_min / 60
    stdev_ct_h = stdev_ct_min / 60
    mu = 1 / charging_time_h
    cv = stdev_ct_h / charging_time_h

    for roh in roh_range:
        lambda_value = roh * (server * mu)

        if roh < 1:
            wq_mmc = (roh / (1 - roh)) * (charging_time_h / server)
            wq_mgc = wq_mmc * ((1 + cv**2) / 2)
            wz_az = wq_mgc / charging_time_h
        else:
            break

        queue.loc[
            lambda_value,
            ["lambda", "server", "roh", "wq_mmc", "wq_mgc", "wz/az", "krit_wert"],
        ] = (
            lambda_value,
            server,
            roh,
            wq_mmc,
            wq_mgc,
            wz_az,
            lambda_value / server,
        )

    return queue.astype("float")


def _qed_servers(lambda_rate, mu, beta=1.0):
    """
    QED (Halfin–Whitt) square-root staffing rule
    """
    R = lambda_rate / mu
    return math.ceil(R + beta * math.sqrt(R))


def _auto_search_radius(lambda_target, mu):
    R = lambda_target / mu
    search_radius = int(round(max(5, 2 * math.sqrt(R)), 0))
    return search_radius


@validate_call
def que_mgc_server_wq_qed(
    lambda_target: Annotated[float, Field(gt=0)],
    charging_time_min: Annotated[int, Field(gt=0)],
    stdev_ct_min: Annotated[int, Field(ge=0)],
    waiting_times_min: list[Annotated[int, Field(gt=0)]],
    method: Literal["coop", "adan", "adan_old"],
    beta: Annotated[float, Field(ge=0)] = 1.0,
    search_radius: Annotated[(int | None), Field(gt=1)] = None,
    max_server: Annotated[int, Field(gt=0)] = 1000,
) -> tuple:
    """
    Determines the required number of servers in an M/G/c queue to serve a target
    arrival rate under mean waiting-time constraints, with optional QED staffing.

    For each target mean waiting time, the function determines the smallest number
    of servers that:
      (i) can stably serve the target arrival rate, and
      (ii) satisfies the specified mean waiting-time constraint.

    The underlying waiting-time evaluation is based on an M/G/c approximation
    (e.g., Cooper or Adan–Resing / Funke). When beta > 0, the search is initialized
    according to the QED (Halfin–Whitt) staffing rule
        c ≈ R + beta * sqrt(R),
    where R = lambda_target / mu is the offered load. The final result, however,
    is determined by feasibility with respect to the waiting-time constraint.

    Parameters
    ----------
    lambda_target : float
        Target arrival rate (λ) in units per hour.

    charging_time_min : int
        Mean service (charging) time in minutes.

    stdev_ct_min : int
        Standard deviation of the service (charging) time in minutes.

    waiting_times_min : list
        List of target mean waiting times (in minutes) for which the required
        number of servers is to be determined.

    method : str
        Queueing approximation used to evaluate mean waiting times.
        Supported options include, for example:
        - 'coop'  : Cooper (1981) M/G/c approximation
        - 'adan'  : Adan & Resing (2017) / Funke (2018) approximation
        - 'adan_old': Adan & Resing (2017) / Funke (2018) approximation with bug in sum_term, kept for comparability

    beta : float, optional
        QED (Halfin–Whitt) quality parameter controlling the amount of safety
        capacity. beta = 0 corresponds to efficiency-driven staffing, while
        beta > 0 introduces additional capacity to improve service quality
        and robustness to variability (default is 1.0).

    search_radius : int, optional
        Numerical search window around the initial server estimate.
        This parameter is used for computational efficiency only and has
        no queueing-theoretic interpretation (default is 10).

    max_server : int, optional
        Maximum number of servers considered in the search (default is 1000).

    Returns
    -------
    tuple
        A tuple consisting of:
        - lambda_target : float
            The target arrival rate.
        - dict_server_wq : dict
            Dictionary mapping each target mean waiting time (in minutes)
            to the corresponding required number of servers. If no feasible
            solution is found within the search range, the value is None.
    """

    dict_method = {
        "coop": queue_mgc_coop,
        "adan_old": queue_mgc_Adan_Resing_old,
        "adan": queue_mgc_Adan_Resing_stable,
    }
    method = dict_method[method]

    dict_server_wq = {}

    charging_time_h = charging_time_min / 60
    stdev_ct_h = stdev_ct_min / 60
    mu = 1 / charging_time_h
    cv = stdev_ct_h / charging_time_h

    # minimum server count for stability
    min_stable_servers = math.ceil(lambda_target / mu)

    if search_radius is None:
        search_radius = _auto_search_radius(lambda_target, mu)
        print(f"Auto search_radius = {search_radius:.1f}")

    for mean_waiting_time_min in waiting_times_min:
        mean_waiting_time_h = mean_waiting_time_min / 60

        # --- QED initial guess ---
        c_qed = _qed_servers(lambda_target, mu, beta)

        search_start = max(min_stable_servers, c_qed - search_radius)
        search_end = min(max_server, c_qed + search_radius)

        # --- Local search around QED (full search, no early break) ---
        feasible_servers = []

        # --- local search around QED ---
        for server in range(search_start, search_end + 1):

            lambda_max, roh_max, wq_mm_c, wq_mg_c, wz_az = method(
                mean_waiting_time_h, server, mu, charging_time_h, cv
            )

            # Server feasible if it can serve lambda_target and meets waiting time
            if lambda_max >= lambda_target:
                feasible_servers.append(server)

        # Pick minimal feasible server to honor QED
        if feasible_servers:
            best_c = min(feasible_servers)
        else:
            best_c = None
            print(
                f"Warning: No server satisfies target Wq={mean_waiting_time_min} min at λ={lambda_target} h⁻¹"
            )

        dict_server_wq[str(mean_waiting_time_min)] = best_c

    return lambda_target, dict_server_wq
