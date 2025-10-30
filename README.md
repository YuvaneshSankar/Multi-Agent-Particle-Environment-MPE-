# MADDPG on Multi-Agent Particle Environment (MPE)

An implementation of Multi-Agent Deep Deterministic Policy Gradient (MADDPG) applied to the
Multi-Agent Particle Environment (MPE). The environment contains multiple agents in a 2D
continuous space performing cooperative and/or competitive tasks.

## Features

- MADDPG algorithm implementation (agents, critics, replay buffer, training loop)
- Environment wrappers for MPE scenarios
- Config-driven training and evaluation

## Repository structure

- `configs/` — YAML configuration files for training and environment parameters.
- `envs/` — Environment wrappers and utilities for MPE scenarios.
- `maddpg/` — Core MADDPG implementation: agents, networks, replay buffer, trainers.
- `scripts/` — Convenience scripts for training and testing.
- `results/` — Logs, tensorboard data, model checkpoints and evaluation outputs.

## Requirements

Python 3.8+ and the Python packages listed in `requirements.txt`.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration

Edit training and environment parameters in `configs/maddpg_mpe_config.yaml` before running.
The config controls learning rates, number of agents, environment name, seed, logging options,
and checkpoint paths.

## Running

Start training with:

```bash
python scripts/train.py --config configs/maddpg_mpe_config.yaml
```

Evaluate a trained model with:

```bash
python scripts/test.py --checkpoint results/checkpoint.pth --config configs/maddpg_mpe_config.yaml
```

Replace the arguments above as needed (different config, checkpoint path, or extra flags supported
by the scripts).

## Quick example

1. Tweak `configs/maddpg_mpe_config.yaml` to your desired settings.
2. Train:

```bash
python scripts/train.py --config configs/maddpg_mpe_config.yaml
```

3. After training, evaluate or visualize results:

```bash
python scripts/test.py --checkpoint results/latest_checkpoint.pth --config configs/maddpg_mpe_config.yaml
```

## Notes

- This repo assumes standard MPE scenarios. If you add custom environments, update `envs/` and the
	corresponding config entries.
- Checkpoints and logs are saved to the path configured in the YAML config (commonly `results/`).

## References

- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)
- Multi-Agent Particle Environment (MPE) by OpenAI

## Contributing

Contributions welcome — please open an issue or a pull request. For major changes, describe
the proposal first so we can discuss design and testing.

## License

Specify your license here (e.g., MIT). If you don't have a license yet, add one to the repository.

---

If you'd like, I can also:

- Add badges (build, license, coverage) to the top of the README.
- Generate a short `CONTRIBUTING.md` and `LICENSE` file.
