import numpy as np
import matplotlib
matplotlib.use("Agg")     #required for HPC so matplotlib doesnt try to open windows
import matplotlib.pyplot as plt
import os
import time
from collections import defaultdict    #creates empty list for new key

os.makedirs("results_hetero_seir", exist_ok=True)       #creates output foolder

print("heterogeneous SEIR script running", flush=True)


seed = 42
rng = np.random.default_rng(seed)          #creates a single random number generator so results can be reproducible


# Real Dublin parameters

N = 592000
density_km2 = 5000

area_km2 = N / density_km2
area_m2 = area_km2 * 1e6
L = np.sqrt(area_m2)


# Simulation parameters

steps = 9000
dt = 0.01

base_move_size = 50.0
infectious_time_mean = 14.0
infectious_time_std = 2.0

latent_time_mean = 4.0
latent_time_std = 1.0

base_p_infect = 0.1
r = 2.0
r2 = r * r
start_infected_frac = 0.0005

# snapshots
snapshot_interval = max(1, steps // 20)   # saves snapshots every 4.5 days (9000/20)


# Heterogeneous population parameters

# 0 = normal, 1 = high-mobility
high_mobility_frac = 0.15

high_move_mult = 1.8
high_infect_mult = 1.35


# Croke Park event

acre = 4046.856
event_area = 12 * acre
event_radius = np.sqrt(event_area / np.pi)   #creating event radius, area of circle formula

event_attendees = 82000
event_center = (L * 0.55, L * 0.55)    #slightly off centre

event_start_day = 2.0
event_duration_day = 0.15

event_start = int(event_start_day / dt)
event_end = event_start + int(event_duration_day / dt)

event_transmission_multiplier = 3.0    #transmission tripled during event


# State definitions

SUSCEPTIBLE = 0
EXPOSED = 1
INFECTIOUS = 2
RECOVERED = 3


# Initialise positions

x = (rng.random(N).astype(np.float32) * L)
y = (rng.random(N).astype(np.float32) * L)


# Initialise agent types, randomly selects 15% of agents and sets theiir type to 1
#ensures no agent is selected twice

agent_type = np.zeros(N, dtype=np.int8)
n_high = int(high_mobility_frac * N)
high_seed_idx = rng.choice(N, n_high, replace=False)
agent_type[high_seed_idx] = 1

#creates per agent multiplier array. agent type, value if true, value if false
move_mult = np.where(agent_type == 0, 1.0, high_move_mult).astype(np.float32)
infect_mult = np.where(agent_type == 0, 1.0, high_infect_mult).astype(np.float32)


# Event attendees  random selection from all agents,
event_indices = rng.choice(N, event_attendees, replace=False)

event_mask = np.zeros(N, dtype=bool)   # true for attendees, false for everyone else
event_mask[event_indices] = True


# Disease state arrays

state = np.zeros(N, dtype=np.int8)   # 0=S, 1=E, 2=I, 3=R
timer = np.zeros(N, dtype=np.float32)

#initial seed infections
n0 = max(1, int(start_infected_frac * N))
start_idx = rng.choice(N, n0, replace=False)
state[start_idx] = INFECTIOUS

#how randomness is implemented for E and I, times are sampled from a normal distribution
def sample_latent_times(size):
    vals = rng.normal(latent_time_mean, latent_time_std, size=size).astype(np.float32)
    return np.clip(vals, 0.5, None)  #prevents unrealistic figures, cant be less than half a day

def sample_infectious_times(size):
    vals = rng.normal(infectious_time_mean, infectious_time_std, size=size).astype(np.float32)
    return np.clip(vals, 1.0, None)    #prevents unrealistic figure, cant be less than a day

timer[start_idx] = sample_infectious_times(n0)   # starts infectious timer for initial seeds


# Storage

t_hist = []
S_hist = []
E_hist = []
I_hist = []
R_hist = []

start_time = time.perf_counter()


# Spatial grid settings

cell_size = r                    # 2 meters
inv_cell = 1.0 / cell_size       #precomputated for speed, helps convert position to cell


#build grid to search for susceptible agents so every S agent doesnt have to be checked
def build_susceptible_grid(x, y, susceptible):
    grid = defaultdict(list)
    gx = (x[susceptible] * inv_cell).astype(np.int32)  #converts postion to cell
    gy = (y[susceptible] * inv_cell).astype(np.int32)

    #check neighbours in the 9 neighbouring grids
    for idx, cx, cy in zip(susceptible, gx, gy):
        grid[(cx, cy)].append(idx)

    return grid


# Output snapshots

def save_snapshots(step, t, x, y, state, t_hist, S_hist, E_hist, I_hist, R_hist, L):
    colors = np.zeros((len(state), 3), dtype=np.float32)
    colors[state == SUSCEPTIBLE] = [0.2, 0.4, 1.0]   # blue
    colors[state == EXPOSED] = [1.0, 0.6, 0.1]       # orange
    colors[state == INFECTIOUS] = [1.0, 0.2, 0.2]    # red
    colors[state == RECOVERED] = [0.2, 0.8, 0.2]     # green

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=1, c=colors)
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_title(f"Dublin Heterogeneous SEIR Snapshot Day {t:.2f}")
    fig.savefig(f"results_hetero_seir/spatial_{step}.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t_hist, S_hist, label="S")
    ax2.plot(t_hist, E_hist, label="E")
    ax2.plot(t_hist, I_hist, label="I")
    ax2.plot(t_hist, R_hist, label="R")
    ax2.legend()
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Population")
    ax2.grid(True, alpha=0.3)
    fig2.savefig(f"results_hetero_seir/seir_{step}.png", dpi=200)
    plt.close(fig2)

    print(f"Saved snapshot at step {step}", flush=True)


# Main loop


for step in range(steps):
    if step % 100 == 0:  #print progress every 100 steps
        print(
            f"step {step}/{steps} | exposed={np.sum(state == EXPOSED)} | infectious={np.sum(state == INFECTIOUS)}",
            flush=True  #force to  terminal
        )

    
    # Movement
    
    if event_start <= step <= event_end:    #if step within event window
        # Move attendees into event area
        angles = rng.random(event_attendees) * 2 * np.pi
        radii = np.sqrt(rng.random(event_attendees)) * event_radius

        #event movement
        x[event_indices] = event_center[0] + radii * np.cos(angles)
        y[event_indices] = event_center[1] + radii * np.sin(angles)

        x[event_indices] %= L   #periodic boundary conditions
        y[event_indices] %= L

        # Non-attendees still move normally using heterogeneous movement
        non_event = ~event_mask  #flips the mask, now true for everyone not at the event
        direction = rng.integers(0, 4, size=np.sum(non_event))   #random number generates normal movement 1=up ... etc
        move_sizes_non_event = base_move_size * move_mult[non_event]

        #normal movement for non event
        x[non_event] = (
            x[non_event]
            + (direction == 1) * move_sizes_non_event
            - (direction == 3) * move_sizes_non_event
        ) % L
        y[non_event] = (
            y[non_event]
            + (direction == 0) * move_sizes_non_event
            - (direction == 2) * move_sizes_non_event
        ) % L

    #movement when no event, high mobility will move more due to multiplier
    else:
        direction = rng.integers(0, 4, size=N)
        move_sizes = base_move_size * move_mult

        x = (x + (direction == 1) * move_sizes - (direction == 3) * move_sizes) % L
        y = (y + (direction == 0) * move_sizes - (direction == 2) * move_sizes) % L

    
    # Infection step
    #gets current indices of all I and S agents 
    infectious = np.where(state == INFECTIOUS)[0]
    susceptible = np.where(state == SUSCEPTIBLE)[0]

    #build S grid
    sus_grid = build_susceptible_grid(x, y, susceptible)
    newly_exposed_chunks = []

    #check if in event window
    in_event_window = event_start <= step <= event_end

    for j in infectious:
        #gets infectious agents grid positions
        cx = int(x[j] * inv_cell)
        cy = int(y[j] * inv_cell)

        #collects all S agents in 9  surrounding cells
        candidates = []

        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                bucket = sus_grid.get((cx + dx_cell, cy + dy_cell))
                if bucket:
                    candidates.extend(bucket)  #if S agent in grid add to candidates

        #if no candidates go to next step
        if not candidates:
            continue

        candidates = np.array(candidates, dtype=np.int32)

        # convert list to array for distance chekc
        dx = x[candidates] - x[j]
        dy = y[candidates] - y[j]
        d2 = dx * dx + dy * dy

        #checks if candidate is in range
        in_range = candidates[d2 < r2]
        if in_range.size == 0:
            continue
        
        #calcualtes transmission probability for this agent
        local_p = base_p_infect * infect_mult[j]
        if in_event_window:
            local_p *= event_transmission_multiplier   #higher transmission if in event window
        local_p = min(local_p, 1.0)
        
        #if random number is less than transmission then they become exposed
        chance = rng.random(in_range.size)
        new_exp = in_range[chance < local_p]

        if new_exp.size > 0:
            newly_exposed_chunks.append(new_exp)   #add newly exposed agents

    if newly_exposed_chunks:
        newly_exposed = np.unique(np.concatenate(newly_exposed_chunks))   #removes duplicates
        newly_exposed = newly_exposed[state[newly_exposed] == SUSCEPTIBLE]  #check theyre still susceptible

        state[newly_exposed] = EXPOSED                                      #change state
        timer[newly_exposed] = sample_latent_times(len(newly_exposed))      #get given latent period timer

    
    # Disease progression
    
    active_mask = (state == EXPOSED) | (state == INFECTIOUS)
    timer[active_mask] -= dt     #lowers all timers by dt 0.01 days

    # E -> I
    exposed_done = np.where((state == EXPOSED) & (timer <= 0))[0]
    if exposed_done.size > 0:
        state[exposed_done] = INFECTIOUS
        timer[exposed_done] = sample_infectious_times(exposed_done.size)

    # I -> R
    infectious_done = np.where((state == INFECTIOUS) & (timer <= 0))[0]
    if infectious_done.size > 0:
        state[infectious_done] = RECOVERED
        timer[infectious_done] = 0.0

    
    # Record totals
    
    t = step * dt
    t_hist.append(t)
    S_hist.append(np.sum(state == SUSCEPTIBLE))
    E_hist.append(np.sum(state == EXPOSED))
    I_hist.append(np.sum(state == INFECTIOUS))
    R_hist.append(np.sum(state == RECOVERED))

    
    # Save snapshots
    
    if step % snapshot_interval == 0 or step == steps - 1:
        save_snapshots(step, t, x, y, state, t_hist, S_hist, E_hist, I_hist, R_hist, L)


# Save final CSV

np.savetxt(
    "results_hetero_seir/dublin_hetero_seir_superspreader_timeseries.csv",
    np.column_stack((t_hist, S_hist, E_hist, I_hist, R_hist)),
    delimiter=",",
    header="time,S,E,I,R",
    comments=""
)

elapsed = (time.perf_counter() - start_time) / 60
print(f"Heterogeneous SEIR simulation complete in {elapsed:.2f} minutes.", flush=True)