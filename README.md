# The Finals – Heavy Class Weapon TTK Modeling (MATLAB)

This repository contains MATLAB scripts for modeling **time-to-kill (TTK)** behavior of Heavy-class weapons.  
The project combines **closed-form weapon math** with **deterministic numerical simulation** to evaluate the impact of spread, pellet dispersion, and range on effective TTK.

No stochastic Monte Carlo sampling is required; all results are deterministic and reproducible.

---

## Repository contents

- `main.m`  
  Entry point for running analyses

- `main_ttk_spread_greyscale_gpu.m`  
  GPU-accelerated deterministic evaluation of TTK across a spread / geometry parameter space

- `main_ttk_spread_differences_gpu.m`  
  GPU-accelerated difference maps comparing two weapon or configuration states

- `ttk_*.m`  
  Weapon-specific scripts defining damage models, cadence, and TTK calculations

---

## Requirements

- MATLAB R20xx or newer  
- Parallel Computing Toolbox (for GPU acceleration)

---

# Mathematical model

The project separates weapon modeling into two layers:

1. **Deterministic weapon math** (damage, cadence, STK, ideal TTK)  
2. **Deterministic spread / pellet simulation** (geometric hit evaluation over parameter grids)

---

## Variables and notation

- $r$ — range  
- $H$ — target health  
- $D(r)$ — base damage per hit at range $r$  
- $m$ — damage multiplier (e.g. headshot); $m=1$ if unused  
- $a$ — damage reduction fraction; $a=0$ if unused  
- $RPM$ — rounds per minute  
- $\Delta t$ — time between shots  
- $M$ — magazine size  
- $T_{reload}$ — reload time  

Effective damage per successful hit:

$$
D_{\text{eff}}(r) = D(r)\cdot m\cdot(1-a)
$$

---

## Damage falloff

Weapons with near and far damage values use a piecewise linear falloff:

$$
D(r)=
\begin{cases}
D_0, & r \le r_0 \\
D_0 + (D_1-D_0)\cdot \frac{r-r_0}{r_1-r_0}, & r_0 < r < r_1 \\
D_1, & r \ge r_1
\end{cases}
$$

If a weapon uses a discrete or non-linear falloff, the corresponding script defines that explicitly.

---

## Shots-to-kill (STK)

### Single-projectile weapons

$$
STK(r) = \left\lceil \frac{H}{D_{\text{eff}}(r)} \right\rceil
$$

---

### Pellet weapons

For weapons firing $N_p$ pellets per shot, each pellet is treated independently.

If $D_p(r)$ is the damage per pellet:

- Damage per pellet hit: $D_p(r)\cdot m\cdot(1-a)$  
- Total damage per shot depends on how many pellets intersect the target

No average pellet assumption is required; pellet hits are evaluated geometrically.

---

## Rate of fire

Time between shots:

$$
\Delta t = \frac{60}{RPM}
$$

---

## Ideal (perfect-accuracy) TTK

If the first shot occurs at $t=0$:

$$
TTK_{\text{ideal}}(r) = (STK(r)-1)\cdot \Delta t
$$

This represents the theoretical lower bound on TTK.

---

## Reload handling (if enabled)

Number of reloads required:

$$
R = \left\lfloor \frac{STK-1}{M} \right\rfloor
$$

Resulting TTK:

$$
TTK(r) = (STK(r)-1)\cdot\Delta t + R\cdot T_{reload}
$$

---

# Spread and pellet simulation (deterministic)

Spread and pellet dispersion are evaluated using **deterministic numerical sampling**, not random trials.

---

## Hit geometry

For a target of radius $R_t$ at range $r$, the angular radius is:

$$
\theta_t \approx \arctan\left(\frac{R_t}{r}\right)
$$

Each shot or pellet is assigned an angular offset drawn from a **deterministic grid** or sweep within the weapon’s spread cone.

A hit occurs if the angular offset lies within $\theta_t$.

---

## Deterministic spread evaluation

Instead of random sampling:

- Spread space is discretized into a fixed grid
- Each grid point is evaluated for intersection with the target
- Hit probability is computed as the fraction of offsets that intersect

This produces:
- exact
- repeatable
- noise-free results

---

## Deterministic pellet evaluation

For pellet weapons:

- Each pellet’s angular offset is evaluated independently
- All possible pellet offsets are sampled deterministically
- Per-shot damage is computed from the resulting pellet hit count

This avoids Monte Carlo variance while preserving spatial structure.

---

## TTK under spread

For each spread configuration:

1. Effective hit rate is computed geometrically
2. Effective damage per shot is derived
3. STK and TTK are computed deterministically

---

# Greyscale spread maps

`main_ttk_spread_greyscale_gpu.m` evaluates TTK over a 2D parameter grid (e.g. range vs spread).

Each pixel represents:
- effective TTK
- expected number of shots
- or kill-time threshold satisfaction

Greyscale intensity corresponds directly to the computed value.

---

# Difference maps (ΔTTK)

`main_ttk_spread_differences_gpu.m` compares two configurations:

$$
\Delta TTK = TTK_B - TTK_A
$$

Interpretation:
- $\Delta TTK < 0$ → configuration B kills faster
- $\Delta TTK > 0$ → configuration A kills faster

---

## Why GPU acceleration is used

Although the simulation is deterministic, it evaluates:
- large parameter grids
- many pellet / angle combinations
- multiple weapons and ranges

GPU parallelism enables exhaustive evaluation without stochastic shortcuts.

---

## How to run

In MATLAB, set the repository as the working directory and run:

```matlab
main
