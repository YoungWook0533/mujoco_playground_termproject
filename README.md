# KWU Reinforce Learning termproject using MuJoCo Playground
Enable a mobile manipulator to learn door-opening skills using the PPO algorithm.
### From Source

> [!IMPORTANT]
> Requires Python 3.10 or later.

1. `cd mujoco_playground_termproject`
2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), a faster alternative to `pip`
3. Create a virtual environment: `uv venv --python 3.11`
4. Activate it: `source .venv/bin/activate`
5. Install CUDA 12 jax: `uv pip install -U "jax[cuda12]"`
    * Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print gpu
6. Install playground: `uv pip install -e ".[all]"`
7. Verify installation (and download Menagerie): `python -c "import mujoco_playground"`

#### Troubleshooting

If, after installing CUDA 12 and JAX, running `python -c "import jax; print(jax.default_backend())"` does not print gpu as the backend, try `unset LD_LIBRARY_PATH` and rerun the command.

#### Madrona-MJX (optional)

For vision-based environments, please refer to the installation instructions in the [Madrona-MJX](https://github.com/shacklettbp/madrona_mjx?tab=readme-ov-file#installation) repository.

## Getting started

### Training
Don't forget to activate virtual environment before running.
```sh
python learning/train_jax_ppo.py --env_name HuskyFR3OpenDoor --run_evals=True --deterministic_rscope=False --use_wandb=True
```
### Visualize results
Pressing reset button in the left panel loads a new initial state.
```sh
python play_policy.py --env_name HuskyFR3OpenDoor --load_checkpoint_path /home/yeonguk/mujoco_playground_termproject/logs/HuskyFR3OpenDoor-final/checkpoints/
```