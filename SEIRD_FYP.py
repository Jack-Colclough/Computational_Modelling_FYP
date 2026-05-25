# SEIRD compartmental model 
# Runs a baseline simulation and parameter sweeps for beta, sigma, mu and I0
# SIR is this script with E and D removed
# SEIR is this script with D removed
 
import numpy as np
import matplotlib.pyplot as plt
 
 
def run_seird(beta, sigma, gamma, mu, S0, E0, I0, R0, D0, N, tmax, dt):
    # total number of timesteps
    steps = int(tmax / dt) + 1
 
    # pre-allocate arrays for each compartment
    S = np.zeros(steps)
    E = np.zeros(steps)
    I = np.zeros(steps)
    R = np.zeros(steps)
    D = np.zeros(steps)
 
    # set initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    D[0] = D0
 
    for t in range(steps - 1):
 
        # rate of new infections — proportional to S*I contact pairs, normalised by N
        inf = beta * S[t] * I[t] / N
 
        # rate of progression from exposed to infectious (mean latent period = 1/sigma days)
        prog = sigma * E[t]
 
        # rate of recovery (mean infectious period = 1/gamma days)
        rec = gamma * I[t]
 
        # rate of death from infectious compartment
        death = mu * I[t]
 
        # forward Euler update: new = old + dt * (inflows - outflows)
        S[t + 1] = S[t] - dt * inf
        E[t + 1] = E[t] + dt * (inf - prog)
        I[t + 1] = I[t] + dt * (prog - rec - death)
        R[t + 1] = R[t] + dt * rec
        D[t + 1] = D[t] + dt * death
 
    time = np.linspace(0, tmax, steps)
    return time, S, E, I, R, D
 
 
def plot_sweep(param_values, param_name, base):
    # run model for each value of one parameter and plot each result 
    for val in param_values:
        p = base.copy()
 
        # changing I0 requires adjusting S0 to keep total population N constant
        if param_name == "I0":
            p["I0"] = val
            p["S0"] = p["N"] - p["E0"] - p["I0"] - p["R0"] - p["D0"]
        else:
            p[param_name] = val
 
        time, S, E, I, R, D = run_seird(
            p["beta"], p["sigma"], p["gamma"], p["mu"],
            p["S0"], p["E0"], p["I0"], p["R0"], p["D0"],
            p["N"], p["tmax"], p["dt"]
        )
 
        plt.figure(figsize=(8, 5))
        plt.plot(time, S, label="S")
        plt.plot(time, E, label="E")
        plt.plot(time, I, label="I")
        plt.plot(time, R, label="R")
        plt.plot(time, D, label="D")
        plt.title(f"SEIRD {param_name} = {val}")
        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
 
 
# baseline parameters
# R0 = beta/gamma = 0.25/0.1 = 2.5
# mean latent period = 1/sigma = 5 days
# mean infectious period = 1/gamma = 10 days
# mortality rate mu = 0.02 per day
params = {
    "beta":  0.25,
    "sigma": 0.2,
    "gamma": 0.1,
    "mu":    0.02,
    "S0": 990,
    "E0": 0,
    "I0": 10,
    "R0": 0,
    "D0": 0,
    "N":    1000,
    "tmax": 160,
    "dt":   0.1
}
 
# baseline run, **params unpacks the dictionary into named arguments
time, S, E, I, R, D = run_seird(**params)
 
plt.figure(figsize=(10, 6))
plt.plot(time, S, label="S")
plt.plot(time, E, label="E")
plt.plot(time, I, label="I")
plt.plot(time, R, label="R")
plt.plot(time, D, label="D")
plt.title("Baseline SEIRD")
plt.xlabel("Days")
plt.ylabel("Population")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
 
# parameter sweeps — one parameter varied at a time, others held fixed
plot_sweep([0.15, 0.25, 0.35, 0.45], "beta",  params)
plot_sweep([0.1,  0.2,  0.3,  0.4],  "sigma", params)
plot_sweep([0.01, 0.02, 0.04, 0.06], "mu",    params)
plot_sweep([1, 10, 30, 50],           "I0",    params)