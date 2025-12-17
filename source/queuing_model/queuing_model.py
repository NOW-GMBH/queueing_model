import pandas as pd
import math
from typing import List, Tuple

def queue_mgc_coop(mean_waiting_time: float, server: int, mu: float, charging_time: float, vk: float,
                   wq_mgc:float=50, roh_0:float=0.99)->list:
    """
    Approximate M/G/c queueing model based on Cooper (1990, p.508, Eq. 9.3).

    Iteratively determines the maximum arrival rate (λ₀) for a multi-server queue
    such that the mean waiting time does not exceed a target value.
    The method uses Cooper’s simplified approximation for M/M/c systems and
    extends it to M/G/c by applying a correction factor based on the coefficient
    of variation of service times (vₖ).

    :param mean_waiting_time: Target mean waiting time in minutes.
    :param server: Number of parallel servers (charging points).
    :param mu: Mean service rate per server [1/hour].
    :param charging_time: Average service (charging) time per customer in hours.
    :param vk: Coefficient of variation of service times (standard deviation / mean).
    :param wq_mgc: Initial waiting time guess (in hours). Default is 50.
    :param roh_0: Initial utilization factor (ρ₀), default is 0.99.
    :return:  list
        [λ₀ (1/h), ρ, Wq_MM_c (min), Wq_MG_c (min), Wq/ServiceTime ratio]
    """

    # After Cooper 1990 - S.508 - Formel 9.3
    lambda_0 = roh_0 * (server * mu)

    while (wq_mgc > mean_waiting_time / 60):
        lambda_0 -= 0.0001
        roh = lambda_0 / (server * mu)
        wq = (roh / (1 - roh)) * (charging_time / server)
        wq_mgc = wq * ((1 + vk ** 2) / 2)
        wz_az = wq_mgc / charging_time

    return [lambda_0, roh, wq * 60, wq_mgc * 60, wz_az]


def queue_mgc_Adan_Resing_old(mean_waiting_time, server, mu, charging_time, vk, wq_mgc=50, roh_0=0.99):
    """
    Warteschlangenmodell im M/G/c-System nach Adan & Resing (2017)

    Approximierung nach Funke 2018 - https://urn.fi/urn:nbn:de:hebis:34-2018041155288 ⎘

    Jedoch falsche Implementierung wq_part2 ist die Summenberechung falsch, sie summiert fälschlicherweise auch den letzten
    Term mit und der Range ist um -1 zu kurz
    :param mean_waiting_time:
    :param server:
    :param mu:
    :param charging_time:
    :param vk:
    :param wq_mgc:
    :param roh_0:
    :return:
    """
    # Adan_Resing
    lambda_0 = roh_0 * (server * mu)

    while (wq_mgc > mean_waiting_time / 60):
        lambda_0 -= 0.0001

        roh = lambda_0 / (server * mu)

        if server > 1:
            wq_part1 = (1 / (1 - roh)) * (1 / (server * mu)) * (((server * roh) ** server) / math.factorial(server))
            wq_part2 = (1 - roh) * sum(
                [(((server * roh) ** n) / math.factorial(n) + ((server * roh) ** server) / math.factorial(server)) for n
                 in range(0, server - 1)])
            wq = wq_part1 * wq_part2 ** -1

        else:
            wq = (roh / (1 - roh)) * (charging_time / server)

        wq_mgc = wq * ((1 + vk ** 2) / 2)
        wz_az = wq_mgc / charging_time

    return [lambda_0, roh, wq * 60, wq_mgc * 60, wz_az]

def queue_mgc_Adan_Resing(mean_waiting_time: float, server: int, mu: float, charging_time: float, vk: float,
                          wq_mgc:float=50, roh_0:float=0.99)->list:
    """

    Calculates the maximum arrival rate for a queueing model using the M/G/c system on the basis of the Adan & Resing (2017)
    and Funke (2018) approoximation.

    Funke (2018) - https://urn.fi/urn:nbn:de:hebis:34-2018041155288

    The function iteratively adjusts the arrival rate (lambda) until the mean waiting time in the queue meets or exceeds the target.
    It uses an approximation method based on Erlang-C formula and adapts it for multiple servers.

    Parameters:
    - mean_waiting_time: Target average waiting time in minutes
    - server: Number of servers available
    - mu: Service rate per server (1 / charging_time)
    - charging_time: Average service time per unit of the system
    - vk: Variations coefficient (standard deviation divided by mean value)
    - wq_mgc: Initial guess for the average waiting time in minutes
    - roh_0: Initial guess for the utilization

    Returns:
    - List containing adjusted lambda, utilization (roh), average waiting time in minutes, and other statistics.
    """
    # --- GEGEBENE GRUNDWERTE ---
    # lambda_0 = roh_0 * (server * mu)
    # wird unten dynamisch angepasst
    # mean_waiting_time in Minuten
    # charging_time = durchschnittliche Bedienzeit (z. B. 0.5 Stunden = 30 Min)
    # mu = 1 / charging_time
    # vk = Variationskoeffizient (Standardabweichung / Mittelwert)

    # ----------------------------------------------
    # START DER ITERATION
    # ----------------------------------------------

    lambda_0 = roh_0 * (server * mu)

    # Schleife: reduziere λ, bis Wartezeit ≤ Ziel
    while wq_mgc > mean_waiting_time / 60:  # mean_waiting_time in Minuten → /60 = Stunden
        lambda_0 -= 0.0001  # Reduzierung der Ankunftsrate

        roh = lambda_0 / (server * mu)

        if roh >= 1:
            # System instabil → überspringen
            continue

        # ----------------------------------------------
        # M/M/c-Formel nach Adan & Resing (2017)
        # ----------------------------------------------
        if server > 1:
            cr = server * roh  # Hilfsgröße

            term = 1.0
            sum_terms = term

            for n in range(1, server):
                term *= cr / n
                # Summe: ∑_{n=0}^{c-1} (cρ)^n / n!
                sum_terms += term

            # last_term = (cr^server)/server! → einfach weitere Iteration
            term *= cr / server
            # Letzter Term: (cρ)^c / c!
            last_term = term

            # Denominator gemäß Paper:
            denom = (1 - roh) * sum_terms + last_term

            # Erlang-C-Wahrscheinlichkeit (P_wait)
            P_wait = last_term / denom

            # M/M/c mittlere Wartezeit Wq_MM_c
            wq = P_wait / (server * mu * (1 - roh))

        # ----------------------------------------------
        # Spezialfall: nur 1 Server → M/G/1 (Pollaczek–Khinchine)
        # ----------------------------------------------
        else:
            E_S = charging_time  # mittlere Bedienzeit
            E_S2 = E_S ** 2 * (1 + vk ** 2)  # zweites Moment über cv
            wq = (lambda_0 * E_S2) / (2 * (1 - roh))  # exakte PK-Formel

        # ----------------------------------------------
        # M/G/c-Approximation nach Funke (2018):
        # Wq_MGc = ((1 + C^2)/2) * Wq_MM_c
        # ----------------------------------------------
        if server > 1:
            wq_mgc = wq * ((1 + vk ** 2) / 2)
        else:
            wq_mgc = wq  # bei M/G/1 schon exakte PK-Formel

        # Verhältnis Wartezeit zu Servicezeit
        wz_az = wq_mgc / charging_time
    return [lambda_0, roh, wq * 60, wq_mgc * 60, wz_az]


def calculate_min_servers(lambda_, mu, max_mean_waiting_time, vk):
    """Adan-Resing"""
    c = 1
    while True:
        rho = lambda_ / (c * mu)
        if rho >= 1:
            c += 1
            continue
        cr = c * rho
        sum_terms = sum((cr ** n) / math.factorial(n) for n in range(0, c))
        last_term = (cr ** c) / math.factorial(c)
        denom = (1 - rho) * sum_terms + last_term
        P_wait = last_term / denom
        wq_mm_c = P_wait / (c * mu * (1 - rho))
        wq_mg_c = ((1 + vk ** 2) / 2) * wq_mm_c
        if wq_mg_c <= max_mean_waiting_time:
            return c, lambda_, rho, wq_mg_c * 60, wq_mm_c * 60,
        c += 1


def _logsumexp(values: List[float]) -> float:
    """Numerisch stabile Berechnung von log(sum(exp(values)))."""
    m = max(values)
    if not math.isfinite(m):
        return float('inf')
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
    if log_ratio > 700:   # exp(700) ~ 1e304
        return 0.0
    if log_ratio < -700:  # exp(-700) ~ 0
        return 1.0

    ratio = math.exp(log_ratio)
    return 1.0 / (1.0 + ratio)


def _compute_wq_for_lambda(
    lmbda: float,
    c: int,
    mu: float,
    charging_time_min: float,
    vk: float
) -> Tuple[float, float, float]:
    """
    Berechne roh, Wq_MM_c (in Minuten) und Wq_MGc (in Minuten)
    für eine gegebene Ankunftsrate λ.

    charging_time_min: Bedienzeit *in Minuten*
    mu: Service rate pro Server = 1 / (charging_time_in_hours)
    """
    # Umrechnung in Stunden für interne Formeln
    charging_time_hr = charging_time_min / 60.0

    if lmbda <= 0:
        return 0.0, 0.0, 0.0

    roh = lmbda / (c * mu)

    if roh >= 1.0:
        return roh, float('inf'), float('inf')

    # -----------------------------
    # M/G/1 Fall
    # -----------------------------
    if c == 1:
        E_S = charging_time_hr
        E_S2 = E_S * E_S * (1 + vk * vk)
        wq_hours = (lmbda * E_S2) / (2 * (1 - roh))
        wq_mgc_hours = wq_hours
        return roh, wq_hours * 60, wq_mgc_hours * 60  # in Minuten

    # -----------------------------
    # M/M/c Basis
    # -----------------------------
    P_wait = _erlang_c_prob_wait(c, roh)
    denom = c * mu * (1 - roh)
    wq_hours = P_wait / denom

    # -----------------------------
    # M/G/c (Funke) Approximation
    # -----------------------------
    wq_mgc_hours = wq_hours * ((1 + vk * vk) / 2.0)

    return roh, wq_hours * 60, wq_mgc_hours * 60  # alles in Minuten


def queue_mgc_Adan_Resing_stable(
    mean_waiting_time: float,
    server: int,
    charging_time_min: float,
    vk: float,
    roh_start: float = 0.99,
    tol_minutes: float = 1e-3,
    max_iter: int = 80
) -> List[float]:
    """
    Stabil berechnete maximale Ankunftsrate lambda für eine Zielwartezeit
    (mean_waiting_time in Minuten).

    Parameter:
    - mean_waiting_time (Min)
    - server = c (Anzahl Server)
    - charging_time_min (Min!)
    - vk = Variationskoeffizient

    Rückgabe:
    [lambda, roh, Wq_MM_c_min, Wq_MGc_min, wz_az]
    """
    # Service rate pro Server in 1/Stunde
    mu = 1.0 / (charging_time_min / 60.0)

    target_wq_min = mean_waiting_time
    target_wq_hr = mean_waiting_time / 60.0

    # Obergrenze für λ: systemstabil knapp unter c*mu
    lambda_high = server * mu * 0.999999
    lambda_low = 0.0

    best_lambda = 0.0
    best_roh = 0.0
    best_wq_min = 0.0
    best_wq_mgc_min = 0.0

    for _ in range(max_iter):
        mid = 0.5 * (lambda_low + lambda_high)

        roh, wq_min, wq_mgc_min = _compute_wq_for_lambda(
            mid, server, mu, charging_time_min, vk
        )

        if not math.isfinite(wq_mgc_min):
            lambda_high = mid
            continue

        if wq_mgc_min <= target_wq_min:
            best_lambda = mid
            best_roh = roh
            best_wq_min = wq_min
            best_wq_mgc_min = wq_mgc_min
            lambda_low = mid
        else:
            lambda_high = mid

        if abs(wq_mgc_min - target_wq_min) < tol_minutes:
            break
        if lambda_high - lambda_low < 1e-12:
            break

    # Verhältnis Warten/Bedienen
    wz_az = best_wq_mgc_min / charging_time_min

    return [
        best_lambda,
        best_roh,
        best_wq_min,
        best_wq_mgc_min,
        wz_az
    ]

def que_mgc(charging_time: int, stdev_ct: int, mean_waiting_time: float, max_server: int, method):
    """
    Calculates the maximum arrival rate for various server counts in a queueing model using different methods.

    This function iterates through a range of server counts and calculates the optimal arrival rate (lambda), traffic intensity (roh),
    average waiting time (wq), maximum queue length (wq_mgc), and other statistics based on the specified method.

    Parameters:
    - charging_time: Average service time in minutes
    - stdev_ct: Standard deviation of the service time in minutes
    - mean_waiting_time: Target average waiting time in minutes for the system
    - max_server: Maximum number of servers to consider (default is 1000)
    - method: Method to use for calculating maximum arrival rate ('coop', 'adan', or 'adan_old')

    Returns:
    - DataFrame containing the calculated parameters for each server count, including the number of servers,
      lambda, roh, wq, wq_mgc, and wz/az.
    """

    dict_method = {'coop': queue_mgc_coop, 'adan': queue_mgc_Adan_Resing, 'adan_old': queue_mgc_Adan_Resing_old}
    method = dict_method[method]

    charging_time = charging_time / 60
    stdev_ct = stdev_ct / 60
    mu = 1 / charging_time
    vk = stdev_ct / charging_time

    queue = pd.DataFrame(0.0, index=list(range(1, max_server + 1)),
                         columns=['servers', 'lambda', 'roh', 'wq', 'wq_mgc', 'wz/az'])

    for server in range(1, max_server + 1):
        queue.loc[server, ['servers', 'lambda', 'roh', 'wq', 'wq_mgc', 'wz/az']] = [server] + method(mean_waiting_time,
                                                                                                     server, mu,
                                                                                                     charging_time, vk)

    return queue


def que_mgc_server_wq(lambda_target: float, charging_time: int, stdev_ct: int, waiting_times: list, method:str,
                      max_server:int=1000)->tuple:
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

    dict_method = {'coop': queue_mgc_coop, 'adan': queue_mgc_Adan_Resing, 'adan_old': queue_mgc_Adan_Resing_old}
    method = dict_method[method]

    dict_server_wq = {}

    charging_time = charging_time / 60
    stdev_ct = stdev_ct / 60
    mu = 1 / charging_time
    vk = stdev_ct / charging_time
    server = 0

    for mean_waiting_time in waiting_times:

        for server in range(1, max_server + 1):

            lambda_0, roh, wq, wq_mgc, wz_az = method(mean_waiting_time, server, mu, charging_time, vk)

            if lambda_0 > lambda_target:
                break

        dict_server_wq[str(mean_waiting_time)] = server

    return lambda_target, dict_server_wq


def queue_wq_roh_coop(roh_range, server, charging_time, stdev_ct):
    queue = pd.DataFrame(columns=['lambda', 'server', 'roh', 'wq', 'wq_mgc', 'wz/az', 'krit_wert'])

    charging_time = charging_time / 60
    stdev_ct = stdev_ct / 60
    mu = 1 / charging_time
    vk = stdev_ct / charging_time

    for roh in roh_range:
        lambda_value = roh * (server * mu)

        if roh < 1:
            wq = (roh / (1 - roh)) * (charging_time / server)
            wq_mgc = wq * ((1 + vk ** 2) / 2)
            wz_az = wq_mgc / charging_time
        else:
            break

        queue.loc[lambda_value, ['lambda', 'server', 'roh', 'wq', 'wq_mgc', 'wz/az',
                                 'krit_wert']] = lambda_value, server, roh, wq * 60, wq_mgc * 60, wz_az, lambda_value / server

    return queue.astype('float')