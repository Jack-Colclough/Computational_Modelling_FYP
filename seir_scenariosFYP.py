# Dublin-scale ABM with heterogeneity and 4 compartments — Intervention Scenarios
# Runs four scenarios to compare the effect of public health interventions
# Scenarios: baseline, vaccination 50%, vaccination 75%, strong distancing
 
 
import numpy as np
import matplotlib
matplotlib.use("Agg")    #required for HPC so plots dont open
import matplotlib.pyplot as plt
import os, csv
from collections import defaultdict   #for spatial grid
 
# SCENARIOS
SCENARIOS = [
    {"name": "01_baseline",            "distancing": 0.0, "vacc_frac": 0.0},
    {"name": "02_vaccination_50pct",   "distancing": 0.0, "vacc_frac": 0.50},
    {"name": "03_vaccination_75pct",   "distancing": 0.0, "vacc_frac": 0.75},
    {"name": "04_strong_distancing",   "distancing": 0.6, "vacc_frac": 0.0},
]
 
 
# FIXED MODEL PARAMETERS
 
N                       = 592000
DENSITY_KM2             = 5000
STEPS                   = 9000
DT                      = 0.01
 
BASE_MOVE_SIZE          = 50.0
BASE_P_INFECT           = 0.1
R                       = 2.0
R2                      = R * R
START_INFECTED_FRAC     = 0.0005
 
INFECTIOUS_TIME_MEAN    = 14.0
INFECTIOUS_TIME_STD     = 2.0
LATENT_TIME_MEAN        = 4.0
LATENT_TIME_STD         = 1.0
 
HIGH_MOBILITY_FRAC      = 0.15
HIGH_MOVE_MULT          = 1.8
HIGH_INFECT_MULT        = 1.35
 
ACRE                    = 4046.856
EVENT_RADIUS            = np.sqrt(12 * ACRE / np.pi)
EVENT_ATTENDEES         = 82000
EVENT_START_DAY         = 2.0
EVENT_DURATION_DAY      = 0.15
EVENT_TRANSMISSION_MULT = 3.0
 
SUS, EXP, INF, REC = 0, 1, 2, 3     #disease states
 
INV_CELL = 1.0 / R                  #for grid cell search
 
#domain size, event location and time
L = np.sqrt((N / DENSITY_KM2) * 1e6)
EVENT_CENTER = (L * 0.55, L * 0.55)
EVENT_START  = int(EVENT_START_DAY / DT)
EVENT_END    = EVENT_START + int(EVENT_DURATION_DAY / DT)
 
#creates output folder if doesnt already exist
OUT_DIR = "results_scenarios"
os.makedirs(OUT_DIR, exist_ok=True)
 
 
# HELPERS
 
#draww duration times from a normal distribution
def sample_latent(rng, size):
    return np.clip(rng.normal(LATENT_TIME_MEAN, LATENT_TIME_STD, size).astype(np.float32), 0.5, None)
 
def sample_infectious(rng, size):
    return np.clip(rng.normal(INFECTIOUS_TIME_MEAN, INFECTIOUS_TIME_STD, size).astype(np.float32), 1.0, None)
 
#build grid for Susceptible search
def build_grid(x, y, agents):
    grid = defaultdict(list)
    gx = (x[agents] * INV_CELL).astype(np.int32)
    gy = (y[agents] * INV_CELL).astype(np.int32)
    for idx, cx, cy in zip(agents, gx, gy):
        grid[(cx, cy)].append(idx)
    return grid
 
 
# SIMULATION FUNCTION
 
def simulate_scenario(name, distancing, vacc_frac, seed=42):
   
    rng = np.random.default_rng(seed)
 
    # effective movement after applying distancing intervention
    eff_move = BASE_MOVE_SIZE * (1.0 - distancing)
 
    # random initial positions for all agents
    x = rng.random(N).astype(np.float32) * L
    y = rng.random(N).astype(np.float32) * L
 
    # assign high-mobility agents — 0 = normal, 1 = high mobility
    agent_type = np.zeros(N, dtype=np.int8)
    n_high     = int(HIGH_MOBILITY_FRAC * N)
    high_idx   = rng.choice(N, n_high, replace=False)
    agent_type[high_idx] = 1
 
    move_mult  = np.where(agent_type == 0, 1.0, HIGH_MOVE_MULT).astype(np.float32)
    inf_mult   = np.where(agent_type == 0, 1.0, HIGH_INFECT_MULT).astype(np.float32)
 
    # select event attendees — random selection from all agents, no attendance bias
    event_indices = rng.choice(N, EVENT_ATTENDEES, replace=False)
 
    event_mask = np.zeros(N, dtype=bool)
    event_mask[event_indices] = True
 
    # initialise disease states
    state = np.zeros(N, dtype=np.int8)
    timer = np.zeros(N, dtype=np.float32)
 
    # vaccination: set a fraction of susceptibles to Recovered (immune) at t=0
    # vaccinated agents start already immune and cannot be infected
    if vacc_frac > 0.0:
        n_vacc   = int(vacc_frac * N)
        vacc_idx = rng.choice(N, n_vacc, replace=False)
        state[vacc_idx] = REC
 
    # seed infections only among susceptibles
    susceptible_pool = np.where(state == SUS)[0]
    n0 = min(max(1, int(START_INFECTED_FRAC * N)), len(susceptible_pool)) # makes sure you cant seed more than the susceptible agents available
    s0 = rng.choice(susceptible_pool, n0, replace=False)
    state[s0] = INF
    timer[s0]  = sample_infectious(rng, n0)   #apply infectious timer
 
    t_hist, S_hist, E_hist, I_hist, R_hist = [], [], [], [], []
 
    for step in range(STEPS):
        in_event = EVENT_START <= step <= EVENT_END
 
        # movement — attendees concentrate at venue during event
        # non-attendees and all agents outside the event window move normally
        if in_event:
            ang = rng.random(EVENT_ATTENDEES) * 2 * np.pi
            rad = np.sqrt(rng.random(EVENT_ATTENDEES)) * EVENT_RADIUS
            x[event_indices] = (EVENT_CENTER[0] + rad * np.cos(ang)) % L
            y[event_indices] = (EVENT_CENTER[1] + rad * np.sin(ang)) % L
 
            non_ev = ~event_mask
            d  = rng.integers(0, 4, size=int(non_ev.sum()))
            ms = eff_move * move_mult[non_ev]
            x[non_ev] = (x[non_ev] + (d==1)*ms - (d==3)*ms) % L
            y[non_ev] = (y[non_ev] + (d==0)*ms - (d==2)*ms) % L
        else:
            d  = rng.integers(0, 4, size=N)
            ms = eff_move * move_mult
            x  = (x + (d==1)*ms - (d==3)*ms) % L
            y  = (y + (d==0)*ms - (d==2)*ms) % L
 
        # infection phase
        infectious  = np.where(state == INF)[0]
        susceptible = np.where(state == SUS)[0]
 
        if infectious.size > 0 and susceptible.size > 0:
            grid   = build_grid(x, y, susceptible)
            chunks = []
 
            for j in infectious:
                cx = int(x[j] * INV_CELL)
                cy = int(y[j] * INV_CELL)
 
                # collect susceptibles from the 9 surrounding grid cells
                cand = []
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        b = grid.get((cx+ddx, cy+ddy))
                        if b:
                            cand.extend(b)
                if not cand:
                    continue
 
                #filter to those within 2 meter radius
                cand = np.array(cand, dtype=np.int32)
                dx2  = (x[cand] - x[j])**2 + (y[cand] - y[j])**2
                close = cand[dx2 < R2]
                if close.size == 0:
                    continue
 
                # transmission probability scaled by infectivity multiplier
                # and event multiplier during the superspreader event
                p = BASE_P_INFECT * inf_mult[j]
                if in_event:
                    p *= EVENT_TRANSMISSION_MULT
                p = min(float(p), 1.0)   # cap at 1.0 so probability cannot exceed 100%
 
                #randomlz decides which nearbz S actually get exposed
                new = close[rng.random(close.size) < p]
                if new.size > 0:
                    chunks.append(new)  #add new to chunks
 
            if chunks:
                newly = np.unique(np.concatenate(chunks)) #removes duplicates
                newly = newly[state[newly] == SUS]         #checks still susceptible
                if newly.size > 0:
                    #change state and start timer
                    state[newly] = EXP
                    timer[newly] = sample_latent(rng, newly.size)
 
        # disease progression
        active = (state == EXP) | (state == INF)
        timer[active] -= DT
 
        # E -> I
        exp_done = np.where((state == EXP) & (timer <= 0))[0]
        if exp_done.size > 0:
            state[exp_done] = INF
            timer[exp_done] = sample_infectious(rng, exp_done.size)
 
        # I -> R
        inf_done = np.where((state == INF) & (timer <= 0))[0]
        if inf_done.size > 0:
            state[inf_done] = REC
            timer[inf_done] = 0.0

        #append new figures
        t_hist.append(step * DT)
        S_hist.append(int(np.sum(state == SUS)))
        E_hist.append(int(np.sum(state == EXP)))
        I_hist.append(int(np.sum(state == INF)))
        R_hist.append(int(np.sum(state == REC)))
 
    # calculate output metrics
    I_arr    = np.array(I_hist)
    R_arr    = np.array(R_hist)
    peak_I   = int(I_arr.max())
    peak_day = float(t_hist[int(I_arr.argmax())])
 
    # attack rate: only count agents who were ever susceptible (exclude vaccinated)
    n_susceptible_at_start = N - int(vacc_frac * N)
    total_R_new = int(R_arr[-1]) - int(vacc_frac * N)   # subtract pre-immune
    attack_rate = total_R_new / max(n_susceptible_at_start, 1)
 
    return {
        "name":        name,
        "t":           t_hist,
        "S":           S_hist,
        "E":           E_hist,
        "I":           I_hist,
        "R":           R_hist,
        "peak_I":      peak_I,
        "peak_day":    peak_day,
        "attack_rate": attack_rate,
    }
 
# PLOTTING
# one colour per scenario
PALETTE = [
    "#e63946",  # 01 baseline          — red
    "#3a86ff",  # 02 vacc 50%          — bright blue
    "#ff006e",  # 03 vacc 75%          — pink
    "#2a9d8f",  # 04 strong distancing — teal
]
 
def plot_all_I_curves(all_results):
    fig, ax = plt.subplots(figsize=(12, 6))
    for res, col in zip(all_results, PALETTE):
        ax.plot(res["t"], res["I"], label=res["name"], color=col, linewidth=1.5)
    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Infectious agents", fontsize=12)
    ax.set_title("Dublin SEIR — Infectious curve by scenario", fontsize=14)
    ax.legend(fontsize=9, ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "all_scenarios_I_curves.png"), dpi=200)
    plt.close(fig)
 
def save_timeseries_csv(res):
    # save S, E, I, R timeseries for this scenario to CSV
    path = os.path.join(OUT_DIR, f"{res['name']}_timeseries.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "S", "E", "I", "R"])
        for row in zip(res["t"], res["S"], res["E"], res["I"], res["R"]):
            writer.writerow(row)
 
 
# MAIN. loops through each scenario saves the csv and creates the compined plot once all runs are completed
if __name__ == "__main__":
    all_results = []
 
    for sc in SCENARIOS:
        res = simulate_scenario(**sc)  #unpacks dictionary
        save_timeseries_csv(res)
        all_results.append(res)
 
    # infectious curve comparison plot across all scenarios
    plot_all_I_curves(all_results)