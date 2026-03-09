# Queuing Model (Team Planen)
> Authors: Matthias Friebel

## About
The queuing model is base for determining the amount of charging points for a given number of charging events per hour
at a specific location for the peak hours. The model is based on a mathematical formula and can be used to predict
how many charging points are needed for different peak hours.

Our queueing model is based on an M/G/c or GI/G/c system. Since no exact closed-form solution exists for the M/G/c
or GI/G/c queue, we apply two approximations in our simulations:

* Allen-Cunneen (GI/G/c): This approximation extends the Lee-Longton approach to systems with non-Poisson arrivals (GI/G/c).
The exact M/M/c waiting time is scaled by the factor $\frac{c_a^2 + C_s^2}{2}$, where $c_a^2$ is the squared coefficient
of variation of interarrival times and $C_s^2$ of service times. For Poisson arrivals ($c_a^2 = 1$) the formula reduces
exactly to the Lee-Longton approximation.
The approximation originates from Allen (1978) and is presented as Formula 9.3 in Cooper (1990, p. 508).

* Lee-Longton (M/G/c): This approximation is based on the Erlang-C formula of the M/M/c system. The exact M/M/c waiting
time is scaled by the factor $\frac{1 + C_s^2}{2}$, taken from the Pollaczek–Khinchine formula, to account for the
variability of service times in an M/G/c system. The approximation originates from Lee and Longton (1959) and has
subsequently been applied in related contexts, such as the modelling of charging infrastructure (e.g. Funke, 2018).

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
