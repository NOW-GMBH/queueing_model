import pandas as pd
import numpy as np
import math


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

            # Summe: ∑_{n=0}^{c-1} (cρ)^n / n!
            sum_terms = sum((cr ** n) / math.factorial(n) for n in range(0, server))

            # Letzter Term: (cρ)^c / c!
            last_term = (cr ** server) / math.factorial(server)

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

    dict_method = {'coop': queue_mgc_coop, 'adan': queue_mgc_Adan_Resing}
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