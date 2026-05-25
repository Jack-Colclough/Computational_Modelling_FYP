# Dublin scale heterogeneous SEIR parameter sweep
# Runs 240 simulations with different parameter combinations
# Parameters are sampled using Latin Hypercube Sampling (LHS)
# Each run saves results to a shared CSV file for neural network training
# Progress is logged to a text file for monitoring during the long HPC run

import numpy as np
import os
import time
import csv
from collections import defaultdict

# create output folder if it does not exist
out_dir = "results_sweep_dublin300"
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(out_dir, "training_data_dublin300.csv")
log_path = os.path.join(out_dir, "progress_log_dublin300.txt")

# log file for tracking progress during long HPC run
def log(msg):
     # print message with timestamp to both terminal and log file
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# fixed model parameters. same for every run in the sweep
N = 592000
density_km2 = 5000
steps = 9000
dt = 0.01
base_move = 50.0
r = 2.5
r2 = r * r

acre = 4046.856
event_radius = np.sqrt(12 * acre / np.pi)
event_start = int(2.0 / dt)
event_end = event_start + int(0.15 / dt)

# disease state constants
SUS = 0
EXP = 1
INF = 2
REC = 3


# parameter ranges for the sweep
# each parameter is varied between its lower and upper bound across the 240 runs
PARAM_BOUNDS = {
    "base_p_infect":               (0.12, 0.35),
    "high_mobility_frac":          (0.05, 0.40),
    "high_move_mult":              (1.2,  3.0),
    "high_infect_mult":            (1.2,  2.5),
    "event_transmission_multiplier": (6.0, 14.0),
    "event_attendees_frac":        (0.08, 0.20),
    "latent_time_mean":            (1.5,  4.5),
    "infectious_time_mean":        (6.0,  18.0),
    "start_infected_frac":         (0.0002, 0.003),
}


def lhs(bounds, n, seed=42):
    
    #Latin Hypercube Sampling across all parameters

    #  For each parameter, the range is divided into n equal intervals (strata).
    # one sample is placed randomly within each stratum.
    # Returns the parameter names and a (n x num_params) array of samples.
    
    rng = np.random.default_rng(seed)
    keys = list(bounds.keys())           # gets parameter names as a list
    samples = np.zeros((n, len(keys)))   # creates 240 x 9 array

    for i, key in enumerate(keys):         #loops through each of the 9  params in turn

        # np.arange(n) creates intervals [0,1,2,...,n-1]
        # adding rng.random(n) places one random point within each interval
        # dividing by n maps everything to [0, 1] with one value per stratum
        strata = (rng.random(n) + np.arange(n)) / n

        # shuffle strata so combinations are not correlated e.g lowest base_p with lowest high_move at [0]
        rng.shuffle(strata)

        # scale from [0,1] to the actual parameter range lo to hi eg. 0.12 to 0.35 for base_p
        lo, hi = bounds[key]
        samples[:, i] = lo + strata * (hi - lo)

    return keys, samples


N_SAMPLES = 240   # number of parameter combinations to test

keys, samples = lhs(PARAM_BOUNDS, N_SAMPLES)


def simulate(params, seed, n=N):
    
    # Run one full Dublin-scale heterogeneous SEIR simulation
    # Takes a parameter dictionary and a random seed
    #Returns epidemic outcome metrics
    
    rng = np.random.default_rng(seed)

    # calculate domain size from population and density
    L = np.sqrt((n / density_km2) * 1e6)
    inv = 1.0 / r   # grid cell

    # random initial positions for all agents
    x = rng.random(n).astype(np.float32) * L
    y = rng.random(n).astype(np.float32) * L

    state = np.zeros(n, dtype=np.int8)
    timer = np.zeros(n, dtype=np.float32)

    # seed initial infections
    n0 = max(1, int(params["start_infected_frac"] * n))
    idx = rng.choice(n, n0, replace=False)
    state[idx] = INF
    timer[idx] = params["infectious_time_mean"]   # fixed per run (not drawn from distribution)

    # assign high-mobility agents 
    agent_type = np.zeros(n, dtype=np.int8)   # 0 = normal, 1 = high mobility
    n_high = max(1, int(params["high_mobility_frac"] * n))
    hi = rng.choice(n, n_high, replace=False)
    agent_type[hi] = 1

    # movement and infectivity multipliers per agent
    # normal agents get multiplier 1.0, high-mobility agents get the sweep value
    move_mult = np.where(agent_type == 0, 1.0, params["high_move_mult"]).astype(np.float32)
    inf_mult  = np.where(agent_type == 0, 1.0, params["high_infect_mult"]).astype(np.float32)

    # select event attendees
    ev_n = max(1, int(params["event_attendees_frac"] * n))
    ev_idx = rng.choice(n, ev_n, replace=False)

    I_hist = []
    R_hist = []

    def build_grid(sus):
        # build spatial grid of susceptible agents for fast neighbour lookup
        g = defaultdict(list)
        gx = (x[sus] * inv).astype(np.int32)
        gy = (y[sus] * inv).astype(np.int32)
        for i_, cx, cy in zip(sus, gx, gy):
            g[(cx, cy)].append(i_)
        return g

    for step in range(steps):
        in_ev = event_start <= step <= event_end

        # movement — each agent moves one step in a random direction
        # high-mobility agents move further because their move_mult is larger
        d  = rng.integers(0, 4, size=n)
        ms = base_move * move_mult   # per-agent step size

        x = (x + (d == 1) * ms - (d == 3) * ms) % L
        y = (y + (d == 0) * ms - (d == 2) * ms) % L

        # event: concentrate attendees within the event circle
        if in_ev:
            ang = rng.random(ev_n) * 2 * np.pi
            rad = np.sqrt(rng.random(ev_n)) * event_radius
            x[ev_idx] = (L * 0.55 + rad * np.cos(ang)) % L
            y[ev_idx] = (L * 0.55 + rad * np.sin(ang)) % L

        # infection phase
        inf_agents = np.where(state == INF)[0]
        sus_agents = np.where(state == SUS)[0]

        if inf_agents.size > 0 and sus_agents.size > 0:
            g = build_grid(sus_agents)
            new_chunks = []

            for j in inf_agents:
                cx = int(x[j] * inv)
                cy = int(y[j] * inv)

                # collect susceptibles from the 9 surrounding grid cells
                cand = []
                for dx_cell in (-1, 0, 1):
                    for dy_cell in (-1, 0, 1):
                        bucket = g.get((cx + dx_cell, cy + dy_cell))
                        if bucket:
                            cand.extend(bucket)

                if not cand:
                    continue

                cand = np.array(cand, dtype=np.int32)
                dx = x[cand] - x[j]
                dy = y[cand] - y[j]
                d2 = dx * dx + dy * dy

                close = cand[d2 < r2]
                if close.size == 0:
                    continue

                # transmission probability for this agent
                # base probability scaled by this agent's infectivity multiplier
                # further scaled by event multiplier during the event
                p = params["base_p_infect"] * inf_mult[j]
                if in_ev:
                    p *= params["event_transmission_multiplier"]
                p = min(float(p), 1.0)   # cap at 1.0 — probability cannot exceed 100%

                new = close[rng.random(close.size) < p]
                if new.size > 0:
                    new_chunks.append(new)

            if new_chunks:
                newly_infected = np.unique(np.concatenate(new_chunks))
                newly_infected = newly_infected[state[newly_infected] == SUS]
                if newly_infected.size > 0:
                    state[newly_infected] = EXP
                    timer[newly_infected] = params["latent_time_mean"]   # fixed per run

        # disease progression
        active = (state == EXP) | (state == INF)
        timer[active] -= dt

        exposed_done = np.where((state == EXP) & (timer <= 0))[0]
        if exposed_done.size > 0:
            state[exposed_done] = INF
            timer[exposed_done] = params["infectious_time_mean"]

        infected_done = np.where((state == INF) & (timer <= 0))[0]
        if infected_done.size > 0:
            state[infected_done] = REC
            timer[infected_done] = 0.0

        I_hist.append(int(np.sum(state == INF)))
        R_hist.append(int(np.sum(state == REC)))

    # calculate output metrics from the completed run
    I = np.array(I_hist)
    R = np.array(R_hist)
    t = np.arange(steps) * dt

    peak_I   = int(I.max())
    peak_day = float(t[I.argmax()])
    total_R  = int(R[-1])
    attack_rate = total_R / n

    # epidemic end day — first step after peak where I drops below 10
    peak_idx = int(I.argmax())
    ends = np.where(I[peak_idx:] < 10)[0]
    epidemic_end_day = float(t[peak_idx + ends[0]]) if ends.size > 0 else float(t[-1])

    # doubling time — how long it takes for I to double from the initial seed count
    early = np.where(I >= n0 * 2)[0]
    doubling_time = float(t[early[0]]) if early.size > 0 else float(t[-1])

    return {
        "peak_I":            peak_I,
        "peak_day":          peak_day,
        "total_R":           total_R,
        "epidemic_end_day":  epidemic_end_day,
        "attack_rate":       attack_rate,
        "doubling_time":     doubling_time,
    }


# main execution. only runs when the script is called directly, not when imported
if __name__ == "__main__":

    # initialise log file
    with open(log_path, "w", encoding="utf-8") as f:        # using w create new file or overwrite previous
        f.write("START\n")

    log(f"Sweep started — {N_SAMPLES} runs, {N} agents per run")

    # open CSV file and write header row
    with open(csv_path, "w", newline="", encoding="utf-8") as f:  #newline stops extra blank lines between rows
        writer = csv.writer(f)
        writer.writerow(["run_id", "seed", *keys,              # *keys list the 9 paramater values
                         "peak_I", "peak_day", "total_R",
                         "epidemic_end_day", "attack_rate",
                         "doubling_time", "elapsed_minutes"])

        run_id = 0                              #number of runs counter
        start  = time.perf_counter()            #record wall time at start

        for i in range(N_SAMPLES):
            # build parameter dictionary for this sample
            params = {keys[k]: float(samples[i, k]) for k in range(len(keys))}

            run_id += 1
            seed = i * 1000   # unique seed per run based on sample index

            #records time before simulation, runs it, then checks how many minutes it took
            t0 = time.perf_counter()
            metrics = simulate(params, seed)
            elapsed = (time.perf_counter() - t0) / 60.0

            # write this run's results immediately — if the job crashes, completed runs are saved
            writer.writerow([
                run_id, seed,
                *[round(params[k], 6) for k in keys],    #rounds to 6 decimal places
                metrics["peak_I"], metrics["peak_day"],
                metrics["total_R"], metrics["epidemic_end_day"],
                metrics["attack_rate"], metrics["doubling_time"],
                round(elapsed, 2),
            ])
            f.flush()   # write to disk immediately so is saved in case of crash

            # calculate and log estimated time remaining
            avg_time  = (time.perf_counter() - start) / run_id
            eta_min   = (avg_time * (N_SAMPLES - run_id)) / 60.0
            log(f"{run_id}/{N_SAMPLES} | peak_I={metrics['peak_I']:,} | "
                f"peak_day={metrics['peak_day']:.2f} | "
                f"time={elapsed:.2f} min | ETA={eta_min:.1f} min")

    total = (time.perf_counter() - start) / 60.0
    log(f"Finished in {total:.2f} minutes")