# The Finals – Heavy Class Weapons Modeling (MATLAB)

This repository models **Time-To-Kill (TTK)** behavior for multiple Heavy-class weapons and generates:
- **Weapon-specific TTK curves** (per gun script)
- **TTK spread/accuracy maps** using GPU Monte Carlo
- **Difference maps** between two configurations/weapons (ΔTTK)

## Files
- `main.m` – entry point (runs chosen analysis)
- `main_ttk_spread_greyscale_gpu.m` – GPU Monte Carlo; outputs a greyscale TTK “spread map”
- `main_ttk_spread_differences_gpu.m` – GPU Monte Carlo; outputs ΔTTK between setups
- `ttk_*.m` – weapon math (damage model + cadence + TTK)

---

## What “TTK” means here

TTK is the time required to deal enough damage to reduce a target from initial health to zero, given:
- weapon damage profile
- rate of fire / cadence
- hit outcome model (perfect accuracy OR probabilistic spread model)
- optional: headshots, multipliers, falloff, pellets, burst logic, reload logic

This repo computes **expected TTK** (or a distribution of TTK) depending on the script.

---

## Core math

### 1) Target health and effective damage
Let:
- `H` = target health (e.g. 250, 300, etc.)
- `m` = damage multiplier (headshot, weakspot, etc.)
- `D(r)` = base damage per hit as a function of range `r` (falloff model)

Effective damage per successful hit:
\[
D_{eff}(r) = D(r)\cdot m
\]

If you use armor / damage reduction, represent it as:
- `a` in `[0,1]` meaning fraction reduced (e.g. 0.2 = 20% reduction)

\[
D_{eff}(r)=D(r)\cdot m\cdot(1-a)
\]

### 2) Damage falloff model (if used)
A common piecewise-linear falloff is:

Given:
- `r0` = no-falloff range
- `r1` = max-falloff range
- `D0` = damage at/under `r0`
- `D1` = damage at/over `r1`

\[
D(r)=
\begin{cases}
D_0, & r\le r_0\\
D_0 + (D_1-D_0)\cdot\frac{r-r_0}{r_1-r_0}, & r_0<r<r_1\\
D_1, & r\ge r_1
\end{cases}
\]

If your scripts use a different falloff (step, exponential, etc.), swap this section to match.

---

## Shots-to-kill (STK)

For single-projectile weapons:
\[
STK(r)=\left\lceil \frac{H}{D_{eff}(r)} \right\rceil
\]

For pellet weapons (shotguns), where each shot fires `N_p` pellets and each pellet deals `D_p(r)`:
- if you assume `k` pellets hit on average per shot:

Damage per shot:
\[
D_{shot}(r)=k\cdot D_p(r)\cdot m
\]

Then:
\[
STK(r)=\left\lceil \frac{H}{D_{shot}(r)} \right\rceil
\]

If you simulate pellet hits directly (Monte Carlo), you don’t use the average `k`; you sample pellet hits per shot.

---

## Rate of fire → time between shots

Let:
- `RPM` = rounds per minute

Time between shots (seconds):
\[
\Delta t = \frac{60}{RPM}
\]

If your weapon fires in bursts, define burst cadence separately (e.g., intra-burst spacing and burst delay).

---

## Deterministic TTK (perfect hits, no reload)

If the first shot is at time 0, and you need `STK` hits, then you need `STK-1` intervals between hits:

\[
TTK_{det}(r) = (STK(r)-1)\cdot \Delta t
\]

This is the baseline “ideal aim” TTK.

---

## Including reload (optional)

Let:
- `M` = magazine size (shots)
- `T_reload` = reload time
- `STK` = shots needed

Number of reloads needed (if you must exceed the magazine):
\[
R = \left\lfloor \frac{STK-1}{M} \right\rfloor
\]

Approximate:
\[
TTK(r) = (STK-1)\Delta t + R\cdot T_{reload}
\]

Notes:
- This assumes reload happens immediately after a shot and blocks further shots.
- If you model partial reload / tactical reload / reload cancel, adapt accordingly.

---

## Probabilistic accuracy / spread modeling (Monte Carlo)

The GPU scripts generate a distribution of TTK outcomes across randomness from spread/recoil.

### 1) Hit probability per shot
For a target of radius `R_t` at range `r`, and an angular spread standard deviation `σ` (radians), the miss/hit geometry can be approximated.

Convert target radius to angular radius:
\[
\theta_t \approx \arctan\left(\frac{R_t}{r}\right)
\]

If you treat shot direction as 2D Gaussian in angle space, the probability that a shot lands within the target disc is:

\[
p_{hit} \approx 1 - \exp\left(-\frac{\theta_t^2}{2\sigma^2}\right)
\]

(There are multiple derivations depending on whether you use Rayleigh radius or 2D normal in polar form; your code may use a different approximation.)

### 2) Stochastic shots-to-kill
If each shot hits with probability `p_hit`, and each hit deals `D_eff`, you can model hits as Bernoulli trials until cumulative damage reaches `H`.

Monte Carlo loop for one trial:
1. time = 0
2. damage = 0
3. while damage < H:
   - sample hit ~ Bernoulli(p_hit)
   - if hit: damage += D_eff
   - if damage >= H: stop
   - time += Δt
4. record time

### 3) Expected TTK and percentiles
From many trials:
- Mean:
\[
\mathbb{E}[TTK]=\frac{1}{N}\sum_{i=1}^N TTK_i
\]
- Median, 10th/90th percentile, etc. computed from the empirical distribution.

---

## Greyscale spread map (what it represents)

In `main_ttk_spread_greyscale_gpu.m`, the output is typically a 2D grid over some pair of variables, for example:
- range vs target size
- range vs spread amount
- range vs aim error
- or two weapon parameters

Each pixel stores either:
- mean TTK
- median TTK
- or probability of killing under a time threshold (depends on your implementation)

Greyscale means darker/lighter corresponds to lower/higher TTK (or vice versa).

---

## Difference map (ΔTTK)

`main_ttk_spread_differences_gpu.m` runs two scenarios A and B and subtracts the results:

\[
\Delta TTK = TTK_B - TTK_A
\]

Interpretation:
- negative ΔTTK: B kills faster than A
- positive ΔTTK: B kills slower than A

This is useful for comparing:
- two weapons
- two balance patches
- two build configs
- two hitbox assumptions

---

## Reproducibility notes
Monte Carlo results depend on RNG seeds. If the code sets a fixed seed, maps are repeatable; if not, expect small run-to-run variance.

---

## Requirements
- MATLAB R20xx+
- Parallel Computing Toolbox (GPU arrays / `gpuArray`)

---

## How to run
Open MATLAB in the repo folder and run:

```matlab
main
