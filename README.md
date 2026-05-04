# Queuing Model (Team Planen)
> Authors: Matthias Friebel

## About
The queuing model determines the number of charging points required to handle a given arrival rate of charging events
per hour at a specific location during peak hours. It is based on a mathematical queueing model and can be applied
across different demand scenarios.

Our model is based on an M/G/c or GI/G/c queueing system. Since no exact closed-form solution exists for either system,
we apply two established approximations:

**Lee-Longton (M/G/c)**: This approximation scales the exact M/M/c mean waiting time ($W_q^{M/M/c}$) by the factor $\frac{1 + C_s^2}{2}$,
taken from the Pollaczek–Khinchine formula, to account for the variability of service times.
The M/M/c waiting time is given by


$W_q^{M/M/c} = \frac{C(c,\, a)}{c\mu - \lambda}$

where $C(c, a)$ is the Erlang-C probability

$C(c, a) = \frac{\dfrac{a^c}{c!} \cdot \dfrac{1}{1 - \rho}}
 {\sum_{k=0}^{c-1} \dfrac{a^k}{k!} + \dfrac{a^c}{c!} \cdot \dfrac{1}{1 - \rho}}$

with offered load $a = \lambda/\mu$ and utilization $\rho = \lambda/(c\mu)$.
The approximation originates from Lee & Longton (1959) and has been applied in related contexts,
including the modelling of charging infrastructure (Funke, 2018).


**Allen-Cunneen (GI/G/c)**: This approximation extends the Lee-Longton approach to systems with non-Poisson arrivals.
The M/M/c waiting time is scaled by the factor $\frac{c_a^2 + C_s^2}{2}$, where $c_a^2$ is the squared coefficient
of variation of interarrival times and $C_s^2$ that of service times. For Poisson arrivals ($c_a^2 = 1$) the formula
reduces exactly to the Lee-Longton approximation. The approximation originates from Allen (1978) and is presented as
Formula 9.3 in Cooper (1990, p. 508).


**Optional — Efficient Server Search:** For large systems, the default
linear search over all feasible server counts can be computationally
expensive. As an alternative, `queue_min_servers_qed` restricts the search
to a focused window around a closed-form anchor estimate:

$$c_{\text{anchor}} = \lceil R + \beta \cdot \sqrt{R} \rceil, \qquad R = \frac{\lambda}{\mu}$$

The final server count is always the smallest `c` within the window that
satisfies the Wq target — the anchor only controls where the search starts,
not the result itself. As a consequence, all results are efficiency-driven:
`queue_min_servers_qed` never returns more servers than strictly necessary
to meet the Wq target.

Setting $\beta = 0$ reduces the anchor to the pure stability bound
$\lceil R \rceil$, which is sufficient for relaxed Wq targets where the
optimal `c` lies close to the stability bound. For stricter targets
(e.g. Wq ≤ 5 min), the optimal `c` lies further above $\lceil R \rceil$ —
a higher $\beta$ ensures the search window is centered closer to the
likely solution.

**Background — QED Staffing Rule:** The anchor formula is borrowed from
the Quality-and-Efficiency-Driven (QED) staffing rule introduced by
Halfin & Whitt (1981). In its original form, the rule uses $\beta > 0$ as
a service quality parameter and returns $c_{\text{anchor}}$ directly as
the final server count — knowingly provisioning more servers than the
minimum in exchange for a statistical waiting time guarantee. In our
implementation, $\beta$ serves a different purpose: it shifts the search
window rather than determining the operating point. The resulting server
count is therefore not a QED operating point, but the efficiency-driven
minimum verified against the Wq target. Users who want an explicit
capacity buffer beyond the minimum can use `_qed_servers` directly:

```python
from queuing_model.queuing_model import _qed_servers

# Returns c_anchor directly — no Wq verification
c_qed = _qed_servers(lambda_target=100, mu=60/20, beta=1.5)
```

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/NOW-GMBH/planen_queuing_model.git
cd queuing-model

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

---

## Usage

The functions `queue_max_lambda`, `queue_min_servers`, `queue_min_servers_qed`, `queue_sweep_rho`,
`queue_sweep_beta`  accept service time and arrival rate parameters in either **minutes** or **hours**
via suffixed keyword arguments:

| Minutes suffix       | Hours suffix          |
|----------------------|-----------------------|
| `charging_time_min`  | `charging_time_hours` |
| `stdev_ct_min`       | `stdev_ct_hours`      |
| `lambda_target_min`  | `lambda_target_hours` |

The output unit for waiting times can be controlled via
the `output_unit` parameter. By default (`output_unit=None`), all waiting
time outputs are retrieved in hours:

| Output field | Default unit   |
|---|----------------|
| `wq` | hour           |
| `lambda` | vehicles / hour |
| `rho` | dimensionless  |

To retrieve raw outputs in **minutes**, set `output_unit="hours_to_minutes"`

### arrival rate λ
The arrival rate λ is used in our models generally in 1/hours and refers to the peak hour of arrivals / charging events.

---

### M/G/c Queue — Lee-Longton Approximation

`queue_mgc_lee_longton` computes the maximum feasible arrival rate λ for a
fixed number of servers and a mean waiting time target.

```python
from queuing_model.queuing_model import queue_mgc_lee_longton

lambda_max, rho, wq_mmc, wq_mgc, wz_az = queue_mgc_lee_longton(mean_waiting_time=5.0 / 60, server=10, mu=60 / 45,
                                                               charging_time=45 / 60, cv=10/45)

print(f"Max arrival rate : {lambda_max:.2f} veh/h")
print(f"Utilization      : {rho:.3f}")
print(f"Wq (M/G/c)       : {wq_mgc * 60:.2f} min")
```

---

### GI/G/c Queue — Allen-Cunneen Approximation

`queue_gigc_allen_cunneen` extends the M/G/c model to non-Poisson arrivals
via the squared coefficient of variation of interarrival times `c_a2`.
Setting `c_a2=1` recovers the Lee-Longton result.

```python
from queuing_model.queuing_model import queue_gigc_allen_cunneen

lambda_max, rho, wq_mmc, wq_gigc, wz_az = queue_gigc_allen_cunneen(mean_waiting_time=5.0 / 60, server=12, mu=60 / 45,
                                                                   charging_time=45 / 60, cv=10 / 45, c_a2=1.5)

print(f"Max arrival rate : {lambda_max:.2f} veh/h")
print(f"Utilization      : {rho:.3f}")
print(f"Wq (GI/G/c)      : {wq_gigc * 60:.2f} min")
```

---

### Maximum Feasible Arrival Rate per Server Count

`queue_max_lambda` sweeps over all server counts up to `max_server` and
returns a DataFrame of maximum feasible arrival rates.

```python
from queuing_model.queuing_model import queue_max_lambda

df = queue_max_lambda(
    charging_time_min=45,
    stdev_ct_min=10,
    wq_target_min=5.0,
    max_server=20,
    method="lee_longton",         # or "allen_cunneen"
)

print(df.head())
#   servers    lambda       roh    wq_mmc    wq_mgc     wz/az
# 1      1.0  0.232991  0.174743  4.999521  4.999521  0.111100
# 2      2.0  1.114745  0.418029  9.528835  4.999697  0.111104
# 3      3.0  2.183836  0.545959  9.528436  4.999488  0.111100
# ...
```

---

### Minimum Server Count

`queue_min_servers` finds the minimum number of servers required to serve
a given arrival rate within a Wq target using a linear search.

```python
from queuing_model.queuing_model import queue_min_servers

lambda_result, server_dict = queue_min_servers(
    lambda_target=10.0,           # target arrival rate [veh/h]
    charging_time_min=45,
    stdev_ct_min=10,
    waiting_times_min=[5.0, 10.0, 20.0, 60.0],
    method="lee_longton",
    output_unit= 'hours_to_minutes'
)

for wq, c in server_dict.items():
    print(f"Wq ≤ {wq} min → {c} servers required")
# Wq ≤ 5.0 min  → 10 servers required
# Wq ≤ 10.0 min →  9 servers required
# ...
```

---

### Minimum Server Count — QED-Guided Search

`queue_min_servers_qed` uses the
[QED staffing rule](https://doi.org/10.1287/opre.29.3.567)
`c = ⌈R + β·√R⌉` to define a focused search window around the likely
optimal server count, which is more efficient for large systems.

```python
from queuing_model.queuing_model import queue_min_servers_qed

lambda_result, server_dict = queue_min_servers_qed(
    lambda_target=100.0,          # target arrival rate [veh/h]
    charging_time_min=20,
    stdev_ct_min=30,
    waiting_times_min=[5.0, 10.0],
    method="lee_longton",
    beta=1.0,                     # Search window center offset. Affects performance only, not the result.
    max_server=500,
    output_unit='hours_to_minutes'
)

for wq, c in server_dict.items():
    print(f"Wq ≤ {wq} min → {c} servers required")
```

> **Note:** For Allen-Cunneen, `c_a2` is required. Omitting it raises a
> `ValidationError`. Setting `c_a2=1.0` reproduces the Lee-Longton result.

### Optional — Sensitivity Analysis:
Two sweep functions are provided to
analyse system behaviour across a range of parameters:

`queue_sweep_rho` computes Wq for a fixed server count and arrival rate
across a range of utilization levels ρ. This is useful for understanding
how sensitive the waiting time is to changes in load — for example to
identify the utilization threshold beyond which Wq deteriorates rapidly.

```python
from queuing_model.queuing_model import queue_sweep_rho

df = queue_sweep_rho(
    servers=10,
    charging_time_min=45,
    stdev_ct_min=10,
    rho_range=[0.5, 0.6, 0.7, 0.8, 0.9],
    method="lee_longton",
)

print(df.head())

#   rho     lambda  servers    wq_mmc    wq_mgc     wz/az
# 0  0.5   6.666667       10  0.005416  0.002842  0.003789
# 1  0.6   8.000000       10  0.018994  0.009966  0.013288
# 2  0.7   9.333333       10  0.055433  0.029085  0.038780
```

`queue_sweep_beta` sweeps over a range of β values and returns the
corresponding server counts and utilization levels for a fixed arrival rate
and Wq target.

```python
from queuing_model.queuing_model import queue_sweep_beta

df = queue_sweep_beta(
    lambda_target=100,
    charging_time_min=20,
    stdev_ct_min=30,
    beta_range=[0.0, 0.5, 1.0, 1.5, 2.0],
    method="lee_longton",
)

print(df.head())

#   beta  servers       rho  lambda    wq_mmc    wq_mgc     wz/az
# 0   0.0       34  0.980392   100.0  0.434305  0.705746  2.117238
# 1   0.5       37  0.900901   100.0  0.039255  0.063790  0.191370
# 2   1.0       40  0.833333   100.0  0.009537  0.015497  0.046492
# 3   1.5       42  0.793651   100.0  0.003957  0.006429  0.019288
# 4   2.0       45  0.740741   100.0  0.001035  0.001682  0.005047

```

Both functions return a `DataFrame` and are intended for exploratory
analysis and parameter tuning rather than production use.

### References:

Allen, A. O. (1978). Probability, Statistics, and Queueing Theory.
        Academic Press.

Cooper, R.B. (1990). Queueing theory. In D.P. Heyman & M.J. Sobel (Eds.),
Handbooks in Operations Research and Management Science (Vol. 2, pp. 469–518).
Elsevier. https://doi.org/10.1016/S0927-0507(05)80174-4

Lee, A.M., Longton, P.A., Queuing Processes Associated with Airline Passenger Check-in. J Oper Res Soc 10(1), (1959),
56–71. doi:10.1057/jors.1959.5.

Pollaczek, F. Über eine Aufgabe der Wahrscheinlichkeitstheorie. I. Math Z 32, 64–100 (1930).
https://doi.org/10.1007/BF01194620

Khintchine, A.Y. (1932). Mathematical theory of a stationary queue. Matematicheski Sbornik, 39, 73–84.

Funke, S.A. (2018). Techno-ökonomische Gesamtbewertung heterogener Maßnahmen zur
Verlängerung der Tagesreichweite von batterieelektrischen Fahrzeugen. (Dissertation, Uni Kassel)

Halfin, S., Whitt, W. (1981). Heavy-traffic limits for queues with many
        exponential servers. Operations Research, 29(3), 567–588.

Borst, S., Mandelbaum, A., Reiman, M. (2004). Dimensioning large
        call centers. Operations Research, 52(1), 17–34.

Gans, N., Koole, G., Mandelbaum, A. (2003). Telephone call centers: Tutorial, review, and research prospects.
 
Manufacturing & Service Operations Management, 5(2), 79–141.
   https://doi.org/10.1287/msom.5.2.79.16071

### Gewährleistung

Der bereitgestellte Code erfolgt ohne Gewähr für Richtigkeit, Vollständigkeit und Aktualität. Die Nutzung erfolgt auf eigene Gefahr.
Eine Haftung für Schäden ist – soweit gesetzlich zulässig – ausgeschlossen. Hiervon unberührt bleibt die Haftung bei Vorsatz, grober Fahrlässigkeit sowie bei Verletzung von Leben, Körper oder Gesundheit.

