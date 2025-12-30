# The-Finals-Heavy-class-weapons-modeling.
# TTK Analysis – MATLAB Scripts

Collection of MATLAB scripts for time-to-kill (TTK) analysis and visualization, including GPU-accelerated spread simulations and weapon-specific models.

## Scripts
| File | Description |
|-----|-------------|
| `main.m` | Entry point to run selected analyses |
| `scripts/main_ttk_spread_greyscale_gpu.m` | GPU-accelerated greyscale TTK spread visualization |
| `scripts/main_ttk_spread_differences_gpu.m` | GPU-based comparison of TTK spread differences |
| `scripts/ttk_M134.m` | TTK model for M134 |
| `scripts/ttk_M60.m` | TTK model for M60 |
| `scripts/ttk_Shak_50.m` | TTK model for ShAK-50 |
| `scripts/ttk_KS_23.m` | TTK model for KS-23 |
| `scripts/ttk_Lewis_Gun.m` | TTK model for Lewis Gun |
| `scripts/ttk_50Akimbo.m` | TTK model for .50 Akimbo |
| `scripts/ttk_BFR_Titan.m` | TTK model for BFR Titan |

## Requirements
- MATLAB R20XX or newer
- Parallel Computing Toolbox (for GPU scripts)

## Usage
1. Clone the repo
2. Open MATLAB and set the repo as the Current Folder
3. Run:
   ```matlab
   main
