from pytest import approx
from queuing_model.queuing_model import queue_mgc_coop

def test_queue_mgc_coop_basic_case():
    mean_waiting_time = 10.0   # min
    server = 4
    mu = 2.0                   # 1/h
    charging_time = 0.5        # h
    vk = 1.0

    lambda_0, roh, wq_mm_c_min, wq_mg_c_min, wz_az = queue_mgc_coop(
        mean_waiting_time=mean_waiting_time,
        server=server,
        mu=mu,
        charging_time=charging_time,
        vk=vk,
    )

    assert lambda_0 > 0.0
    assert 0.0 < roh < 1.0
    assert wq_mg_c_min <= mean_waiting_time
    assert wz_az > 0.0

    # Mit pytest.approx für Float-Toleranz
    assert (wq_mm_c_min / 60) == approx((roh / (1 - roh)) * (charging_time / server), rel=1e-6)