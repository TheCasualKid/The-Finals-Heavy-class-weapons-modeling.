# Expected TTK (Time-To-Kill) Simulator (MATLAB, GPU)

This repo computes **expected / average Time-To-Kill (TTK)** for multiple weapons over a 3D grid of:

- `q` = accuracy = probability a shot hits (`0..1`)
- `h` = headshot fraction = probability a hit is a headshot (`0..1`)
- `dist_m` = distance in meters (`>=0`)

It evaluates each weapon’s **expected TTK** and then visualizes:
- which weapon is best (Rank #1 / #2 / #3),
- and how *far apart* the top-ranked weapons are (TTK spreads).

---

## Repo structure

- `main.m`
  - Computes expected TTK for every `ttk_*.m` weapon across the grid
  - Produces 3 plots: Rank #1, #2, #3 weapon at each `(q,h,dist)`
- `main_ttk_spread_greyscale_gpu.m`
  - Computes **Δ12 = TTK(rank2)-TTK(rank1)** and **Δ23 = TTK(rank3)-TTK(rank2)**
  - Plots both as greyscale points where **1 second = fully black**
- `main_ttk_spread_differences_gpu.m`
  - Same deltas as above, but uses a **cap** (default 2 seconds) for black
- `ttk_*.m`
  - One file per weapon. Each returns **expected TTK** (first output), GPU-vectorized.

---

## The math: expected TTK from per-shot outcomes

### 1) Per-shot outcome probabilities

Each shot has three mutually exclusive outcomes:

- **Miss**: probability `pM = 1 - q`
- **Body hit**: probability `pB = q (1 - h)`
- **Head hit**: probability `pH = q h`

So `pM + pB + pH = 1`.

### 2) Damage with distance falloff

Each weapon defines a distance multiplier `m(dist)` as a **piecewise function** (different per gun).
The code typically discretizes distance in millimeters (`mm = round(...)` or `mm = floor(...)`) to exactly match your Python model.

Example pattern used in several guns:
- `m = 1` up to some range,
- linearly decreases to a minimum,
- then clamps at a floor (e.g. `0.5`, `0.4`, etc.)

Effective damages:
- `D_body(dist) = D_body_base * m(dist)`
- `D_head(dist) = D_head_base * m(dist)`

### 3) Integer “unit-space” conversion (key trick)

To make the probability DP exact and GPU-friendly, each weapon converts damage into **integer steps**.

You pick a unit size so that:
- body damage is an integer number of units (`BODY_STEP`)
- head damage is an integer number of units (`HEAD_STEP`)
- and the kill threshold is an integer `T` (units)

Then the fight becomes: accumulate units until total ≥ `T`.

Examples from your code:
- M60 / ShAK / Akimbo / etc use small unit steps like `BODY_STEP=2`, `HEAD_STEP=3`
- M134 uses large steps `BODY_STEP=100`, `HEAD_STEP=133` to represent 11 and 14.63 exactly

Threshold `T` is computed as a ceiling, so the model kills as soon as damage meets/exceeds HP.

### 4) Distribution of “alive” states after each shot (DP)

Let `U_n` be the total accumulated damage-units after `n` shots.

We track the probability distribution of `U_n` **only for alive states**: `U_n < T`.

Define:
- `dp_n[u] = P(U_n = u AND u < T)`

Initialization (before any shot):
- `dp_0[0] = 1`, all others 0.

Transition for each shot:
- Miss: stays at same unit `u`
- Body: shifts by `+BODY_STEP`
- Head: shifts by `+HEAD_STEP`

For each step `n -> n+1`, conceptually:
- `dp_{n+1}[u] += dp_n[u] * pM`
- `dp_{n+1}[u+BODY_STEP] += dp_n[u] * pB`
- `dp_{n+1}[u+HEAD_STEP] += dp_n[u] * pH`

Then we **truncate** any state `u >= T` (dead states are not stored).

In the code this is implemented by masking:
- `aliveMask[u] = (u < T)`
- `dp = dp .* aliveMask` after each update

### 5) Tail probability (probability still alive after n shots)

After computing `dp_n`, the probability the target is still alive is:
- `tail[n] = P(alive after n) = sum_u dp_n[u]`

So:
- `tail[0] = 1`
- `tail[n]` is non-increasing in `n`

### 6) Expected bullets to kill: capped vs uncapped

Let `τ` be the (random) shot index where the kill happens.

A standard identity:
\[
E[\tau] = \sum_{n=0}^{\infty} P(\tau > n) = \sum_{n=0}^{\infty} P(\text{alive after } n)
\]

#### A) Within one magazine (capped at `MAG`)
Your gun functions compute:
\[
E[\min(\tau, MAG)] = \sum_{n=0}^{MAG-1} P(\text{alive after } n)
\]
In code this is accumulated as:
- start `EbMag = tail[0]`
- for `n=1..MAG-1`, add `tail[n]`

Also computed:
\[
pKill = P(\tau \le MAG) = 1 - P(\text{alive after } MAG) = 1 - tail[MAG]
\]

So each weapon can return:
- `pKill` = probability you kill within one magazine
- `EbMag` = expected **capped** bullets fired in that magazine

#### B) Uncapped (multiple magazines, empty reloads)
Several weapons then convert the single-mag results into an “infinite fight” model.

Assumption used in your files: **EMPTY reload only** (no tactical reload mid-mag), and you always continue with fresh mags until the kill occurs.

A clean identity used in your code:
\[
E[\tau] = \frac{E[\min(\tau, MAG)]}{pKill}
\]
and expected number of empty reloads:
\[
E[\#reloads] = \frac{1 - pKill}{pKill}
\]
(geometric “success per mag” model)

In code:
- `EbUncapped = EbMag / pKill`
- `EReloads = (1 - pKill) / pKill`

If `pKill = 0`, expected time is `Inf`.

### 7) Expected TTK in seconds

Let:
- `RPM` = rounds per minute
- `dt = 60 / RPM` = seconds between shots
- shots occur at times `0, dt, 2dt, ...`

So if you fire `k` bullets, the time of the `k`-th shot is `(k-1)dt`.

#### Capped TTK (single-mag model)
Used for M134:
\[
E[TTK] = \text{START\_DELAY} + dt \cdot (E[\min(\tau, MAG)] - 1)
\]

#### Uncapped TTK (multi-mag model with empty reloads)
Used for M60/ShAK/KS23/Akimbo/BFR/Lewis etc:
\[
E[TTK] = dt \cdot (E[\tau] - 1) + \text{EMPTY\_RELOAD}\cdot E[\#reloads]
\]
Some guns also add a fixed startup delay if the model requires it.

---

## What each weapon file does

All weapon files follow the same structure:

1) Validate inputs (`q,h` in `[0,1]`, `dist_m >= 0`)  
2) Broadcast inputs to common array sizes (so `q,h,dist_m` can be grids)  
3) Compute distance multiplier `m(dist)` using mm discretization  
4) Compute kill threshold `T` in integer unit-space  
5) Run DP for `n = 1..MAG` to get `tail[n]`, `pKill`, and `EbMag`  
6) Optionally convert to uncapped `EbUncapped` and expected reloads  
7) Convert expected bullets -> expected time seconds

### Specific notes from your code

- `ttk_M134.m`
  - **MAG=250**, **RPM=1500**
  - Includes **START_DELAY** (spin-up)
  - Uses **chunking** (`CHUNK_M`) because DP state size is huge (`Tmax` ~ 7955)
  - Returns capped expectation: `E[min(τ,250)]`

- `ttk_M60.m`
  - **MAG=70**, **RPM=580**
  - Falloff: `<=25m => 1`, `25-35m` linear to `0.5`, `>=35m => 0.5`
  - Uses uncapped conversion + **EMPTY reload** time (3.55s in-file)
  - Returns: `Ettk, pKill, EbMag, EbUncapped`

- `ttk_Shak_50.m`, `ttk_50Akimbo.m`, `ttk_Lewis_Gun.m`, `ttk_BFR_Titan.m`
  - Same “uncapped with empty reloads” pattern (different stats and falloff)

- `ttk_KS_23.m`
  - Slug-style model (body only; `h` isn’t used for damage outcome)
  - Low RPM, small MAG, long reload → uncapped reload term matters a lot

---

## What the main scripts do

### `main.m` (Ranked best guns plot)
- Builds grid:
  - `q_vals = linspace(0.10, 1.00, 46)`
  - `h_vals = linspace(0.00, 1.00, 46)`
  - `dist_vals = linspace(0, 50, 51)`
- Finds all `ttk_*.m` files automatically
- Evaluates each gun on GPU (optional), storing `TTK(point, gun)`
- Computes Rank #1/#2/#3 **without sorting** (min, then mask best as `inf`, min again, etc.)
- Plots 3 scatter3 plots:
  - points colored by which gun is Rank #1, then Rank #2, then Rank #3

### `main_ttk_spread_greyscale_gpu.m` (Spread plots, 1s black)
- Computes best 3 TTK values (`ttk1, ttk2, ttk3`)
- Calculates spreads:
  - `d12 = ttk2 - ttk1`
  - `d23 = ttk3 - ttk2`
- Maps spread to greyscale with:
  - `greyness = min(Δt / 1.0, 1)` → 0 white, 1 black
- Produces 2 scatter3 plots (Δ12, Δ23)

### `main_ttk_spread_differences_gpu.m` (Spread plots, capped at 2s)
- Same spreads as above
- Uses a `diffCap` (default **2.0 seconds**) so:
  - 0 → white, ≥2 → black
- Produces 2 scatter3 plots with a colorbar in seconds

---

## Adding a new gun

1) Create a new file: `ttk_MyGun.m`
2) Function signature should accept:
```matlab
function Ettk = ttk_MyGun(q, h, dist_m)
