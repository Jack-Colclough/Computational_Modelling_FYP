# superspreaderSEIR.py
# Dublin-scale heterogeneous SEIR agent-based simulation with 592,000 agents 
# agents infect neighbours within radius r
# Includes a superspreader event (Croke Park) at day 2
# Each agent has an individual latent and infectious period drawn from normal distributions
# Results saved as spatial snapshots, SEIR timeseries plots and a CSV

import numpy as np
import matplotlib
matplotlib.use("Agg")   # stops display window, writes directly to file
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict

os.makedirs("results_seir", exist_ok=True)


# Dublin population and domain size
# density of 5000 people/km2 gives domain area = N/density = 118.4 km2
# L is the side length of the square domain in metres

N = 592000
density_km2 = 5000
area_km2 = N / density_km2
area_m2 = area_km2 * 1e6
L = np.sqrt(area_m2)   #approx 10,840 metres for sides of square


# Simulation parameters

steps = 9000        # total timesteps
dt = 0.01          # each step = 0.01 days so 9000 steps = 90 days

move_size = 50.0   # metres per step , one random walk step in one of four directions

# disease duration parameters. Each agent gets their own value drawn from these distributions
infectious_time_mean = 14.0
infectious_time_std  = 2.0
latent_time_mean     = 4.0
latent_time_std      = 1.0

p_infect = 0.1    # probability of transmission per timestep if within radius r
r  = 2.0          # transmission radius in metres
r2 = r * r        # squared radius — used to avoid sqrt in distance calculations

start_infected_frac = 0.0005   # fraction of population infectious at t=0

# fixed random seed ensures results are reproducible
seed = 42
rng = np.random.default_rng(seed)


# Superspreader event — modelled on Croke Park (12 acres)
# 82,000 attendees are concentrated within the event radius for 0.15 days

acre = 4046.856
event_area   = 12 * acre
event_radius = np.sqrt(event_area / np.pi)   # circular event area

event_attendees = 82000
event_center    = (L * 0.55, L * 0.55)       # event location within the domain

event_start_day  = 2.0
event_duration_day = 0.15


# convert event timing from days to simulation steps
event_start = int(event_start_day / dt)
event_end   = event_start + int(event_duration_day / dt)

# randomly select which agents attend, same agents every time due to fixed seed, replace false means same agent cant be selected twice
event_indices = rng.choice(N, event_attendees, replace=False)


# Disease state definitions 
SUSCEPTIBLE = 0
EXPOSED     = 1
INFECTIOUS  = 2
RECOVERED   = 3


# Initial random agent positions within domain , memory halved using float32
x = rng.random(N).astype(np.float32) * L
y = rng.random(N).astype(np.float32) * L


# both state and timer start at 0. everyone susceptible
# state array: one integer per agent (0,1,2,3)
state = np.zeros(N, dtype=np.int8)

# timer array: stores remaining latent or infectious time per agent
timer = np.zeros(N, dtype=np.float32)

# seed initial infections
n0 = max(1, int(start_infected_frac * N))
start_idx = rng.choice(N, n0, replace=False)
state[start_idx] = INFECTIOUS

# assign individual infectious periods to the initially infected agents
init_times = rng.normal(infectious_time_mean, infectious_time_std, size=n0).astype(np.float32)
init_times = np.clip(init_times, 1.0, None)   # minimum 1 day infectious
timer[start_idx] = init_times


# Storage for SEIR timeseries
t_hist = []
S_hist = []
E_hist = []
I_hist = []
R_hist = []

snapshot_interval = steps // 20   # save a snapshot every 450 steps (every 4.5 days)
start_time = time.perf_counter()


# Spatial grid for fast neighbour lookup
# Instead of checking every agent against every other (N^2 pairs), the domain is divided into cells of size r x r
# Each infectious agent only checks the 9 cells around it for susceptibles.


cell_size = r                   # cell size is transmission radius
inv_cell  = 1.0 / cell_size   # multiply is faster than divide


def sample_latent_times(size):
    # draw individual latent periods from N(mean, std), clip at 0.5 days minimum
    vals = rng.normal(latent_time_mean, latent_time_std, size=size).astype(np.float32)
    return np.clip(vals, 0.5, None)

def sample_infectious_times(size):
    # draw individual infectious periods from N(mean, std), clip at 1 day minimum
    vals = rng.normal(infectious_time_mean, infectious_time_std, size=size).astype(np.float32)
    return np.clip(vals, 1.0, None)

def save_snapshots(step, t, x, y, state, t_hist, S_hist, E_hist, I_hist, R_hist, L):
    # colour agents by disease state
    colors = np.zeros((N, 3), dtype=np.float32)
    colors[state == SUSCEPTIBLE] = [0.2, 0.4, 1.0]   # blue
    colors[state == EXPOSED]     = [1.0, 0.6, 0.1]   # orange
    colors[state == INFECTIOUS]  = [1.0, 0.2, 0.2]   # red
    colors[state == RECOVERED]   = [0.2, 0.8, 0.2]   # green

    # spatial snapshot — one dot per agent
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=1, c=colors)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_title(f"Dublin SEIR Superspreader Snapshot Day {t:.2f}")
    fig.savefig(f"results_seir/spatial_{step}.png", dpi=200)
    plt.close(fig)

    # SEIR timeseries up to this point
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t_hist, S_hist, label="S")
    ax2.plot(t_hist, E_hist, label="E")
    ax2.plot(t_hist, I_hist, label="I")
    ax2.plot(t_hist, R_hist, label="R")
    ax2.legend()
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Population")
    ax2.grid(True, alpha=0.3)
    fig2.savefig(f"results_seir/seir_{step}.png", dpi=200)
    plt.close(fig2)

    print(f"Saved snapshot at step {step}", flush=True)

def build_susceptible_grid(x, y, susceptible):
    # build a dictionary mapping each grid cell to the list of susceptible agents in it
    # key = (cell_x, cell_y), value = list of agent indices
    grid = defaultdict(list)
    gx = (x[susceptible] * inv_cell).astype(np.int32)
    gy = (y[susceptible] * inv_cell).astype(np.int32)
    for idx, cx, cy in zip(susceptible, gx, gy):
        grid[(cx, cy)].append(idx)
    return grid


# Main simulation loop

for step in range(steps):

    # progress update every 100 steps
    if step % 100 == 0:
        print(f"step {step}/{steps} | E={np.sum(state==EXPOSED)} | I={np.sum(state==INFECTIOUS)}", flush=True)


    # Movement phase

    if event_start <= step <= event_end:
        # during the event, attendees are placed randomly within the event circle
        # uniform distribution over a disc: radius scaled by sqrt of random number
        angles = rng.random(event_attendees) * 2 * np.pi
        radii  = np.sqrt(rng.random(event_attendees)) * event_radius
        x[event_indices] = (event_center[0] + radii * np.cos(angles)) % L
        y[event_indices] = (event_center[1] + radii * np.sin(angles)) % L
    else:
        # normal movement: each agent moves one step in a random direction
        # direction 0=up, 1=right, 2=down, 3=left
        # % L applies periodic boundary conditions — agents wrap around the domain edges
        direction = rng.integers(0, 4, size=N)
        x = (x + (direction == 1) * move_size - (direction == 3) * move_size) % L
        y = (y + (direction == 0) * move_size - (direction == 2) * move_size) % L


    # Infection phase
    # Are tuples [0] returns just the array
    infectious  = np.where(state == INFECTIOUS)[0]
    susceptible = np.where(state == SUSCEPTIBLE)[0]

    # transmission probability is tripled during the event (confined space)
    local_p = p_infect * 3.0 if event_start <= step <= event_end else p_infect

    # build the spatial grid of susceptible agents for fast lookup
    # rebuild every step cause agents have moved
    sus_grid = build_susceptible_grid(x, y, susceptible)
    newly_exposed_chunks = []

    for j in infectious:
        # find which grid cell this infectious agent is in
        cx = int(x[j] * inv_cell)
        cy = int(y[j] * inv_cell)

        # collect susceptible agents from the 9 surrounding cells (3x3 neighbourhood)
        candidates = []
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                bucket = sus_grid.get((cx + dx_cell, cy + dy_cell))   # returns none if key doesnt exist
                if bucket:
                    candidates.extend(bucket)      # adds all susceptibles from cell to candidates list

        if not candidates:                        # if candidates list is empty skip on
            continue

        # calculate exact distances to candidate susceptibles
        candidates = np.array(candidates, dtype=np.int32)
        dx = x[candidates] - x[j]
        dy = y[candidates] - y[j]
        d2 = dx * dx + dy * dy   # squared distance 

        # keep only those within transmission radius r
        in_range = candidates[d2 < r2]
        if in_range.size == 0:
            continue

        # each agent in range is exposed with probability local_p
        # creates random number in range, if less than local_p, becomes infected
        chance  = rng.random(in_range.size)
        new_exp = in_range[chance < local_p]

        #if number of newly exposed is greater than 0 add newly exposed to newly exposed chunks
        if new_exp.size > 0:
            newly_exposed_chunks.append(new_exp)

    # apply all new exposures at once
    if newly_exposed_chunks:
        newly_exposed = np.unique(np.concatenate(newly_exposed_chunks))
        # double check state is still S two infectious agents could have targeted the same person, so np.unique removes duplicates
        newly_exposed = newly_exposed[state[newly_exposed] == SUSCEPTIBLE]
        state[newly_exposed] = EXPOSED
        timer[newly_exposed] = sample_latent_times(len(newly_exposed))   # individual latent periods


    # Disease progression phase

    # count down timers for all exposed and infectious agents
    active_mask = (state == EXPOSED) | (state == INFECTIOUS)
    timer[active_mask] -= dt

    # E to I: exposed agents whose latent period has expired become infectious
    exposed_done = np.where((state == EXPOSED) & (timer <= 0))[0]
    if exposed_done.size > 0:
        state[exposed_done] = INFECTIOUS
        timer[exposed_done] = sample_infectious_times(exposed_done.size)   # new individual infectious periods

    # I to R: infectious agents whose infectious period has expired recover
    infectious_done = np.where((state == INFECTIOUS) & (timer <= 0))[0]
    if infectious_done.size > 0:
        state[infectious_done] = RECOVERED
        timer[infectious_done] = 0.0


    # Record compartment counts for this timestep
    t = step * dt
    t_hist.append(t)
    S_hist.append(np.sum(state == SUSCEPTIBLE))
    E_hist.append(np.sum(state == EXPOSED))
    I_hist.append(np.sum(state == INFECTIOUS))
    R_hist.append(np.sum(state == RECOVERED))

    # save spatial and timeseries snapshots at regular intervals
    if step % snapshot_interval == 0 or step == steps - 1:
        save_snapshots(step, t, x, y, state, t_hist, S_hist, E_hist, I_hist, R_hist, L)


# Save full timeseries to CSV for analysis
np.savetxt(
    "results_seir/dublin_seir_superspreader_timeseries.csv",
    np.column_stack((t_hist, S_hist, E_hist, I_hist, R_hist)),     #takes five lists and combines them into an array
    delimiter=",",
    header="time,S,E,I,R",
    comments=""           #stops numpy adding #, breaking the csv
)

elapsed = (time.perf_counter() - start_time) / 60
print(f"Simulation complete in {elapsed:.2f} minutes.", flush=True)