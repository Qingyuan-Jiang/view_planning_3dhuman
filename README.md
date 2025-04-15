# View Planning for High-Fidelity Reconstruction.

This repo is combined with the paper "View Planning for High-Fidelity Reconstruction of Dynamic Actor using Flying Camera".

**Please check out our [supplementary video](https://youtu.be/bzMjb8sJFgs).**

## Installation
Install in the virtual environments, and the API for the AirSim with `pip`.

```pip install airsim```

## Codes structure
Environments are placed in the `envs` folder. The `envs` folder contains the following files:
- `vp_drone.py' : wrapper of APIs for the drone
- `env_airsim.py': the environment for the simulation

To use the environment, you may want to call scripts from `simulation` folder.
- 'sim_vp_local.py': call the environment in CV model.

View planning algortihm based on Pixels-Per-Area (PPA) metric is placed in `vp_local` folder.
- 'vp_planner.py`: compute PPA values and optimize view points based on PPAs.
