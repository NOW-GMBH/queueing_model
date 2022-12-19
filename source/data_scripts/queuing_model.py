import pandas as pd
import numpy as np
import math


def queue_mgc_coop(mean_waiting_time, server, mu, charging_time, vk, wq_mgc=50, roh_0=0.99):
    # After Coop 1990 - S.508 - Formel 9.3
    lambda_0 = roh_0 * (server * mu)

    while (wq_mgc > mean_waiting_time / 60):
        lambda_0 -= 0.0001
        roh = lambda_0 / (server * mu)
        wq = (roh / (1 - roh)) * (charging_time / server)
        wq_mgc = wq * ((1 + vk ** 2) / 2)
        wz_az = wq_mgc / charging_time

    return [lambda_0, roh, wq * 60, wq_mgc * 60, wz_az]


def queue_mgc_Adan_Resing(mean_waiting_time, server, mu, charging_time, vk, wq_mgc=50, roh_0=0.99):
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


def que_mgc(charging_time, stdev_ct, mean_waiting_time, max_server, method):
    dict_method = {'coop': queue_mgc_coop, 'adan': queue_mgc_Adan_Resing}
    method = dict_method[method]

    charging_time = charging_time / 60
    stdev_ct = stdev_ct / 60
    mu = 1 / charging_time
    vk = stdev_ct / charging_time

    queue = pd.DataFrame(0, index=list(range(1, max_server + 1)),
                         columns=['servers', 'lambda', 'roh', 'wq', 'wq_mgc', 'wz/az'])

    for server in range(1, max_server + 1):
        queue.loc[server, ['servers', 'lambda', 'roh', 'wq', 'wq_mgc', 'wz/az']] = [server] + method(mean_waiting_time,
                                                                                                     server, mu,
                                                                                                     charging_time, vk)

    return queue


def que_mgc_server_wq(lambda_target, charging_time, stdev_ct, waiting_times, method, max_server=1000):
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