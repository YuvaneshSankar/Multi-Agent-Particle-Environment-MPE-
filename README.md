# MADDPG on Multi-Agent Particle Environment (MPE)

This project implements the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm on the Multi-Agent Particle Environment (MPE). The environment involves multiple agents interacting in a 2D continuous space, performing cooperative or competitive tasks.

## Project Structure

- `configs/` - Configuration files for training and environment parameters.
- `envs/` - Environment wrappers for MPE scenarios.
- `maddpg/` - MADDPG core algorithm implementation including agents, replay buffer, models, and training.
- `scripts/` - Scripts to run training and evaluation.
- `results/` - Logs, model checkpoints, and results.

## Setup

1. Clone the repo.
2. Install dependencies:

pip install -r requirements.txt

text

3. Adjust the training and environment parameters in `configs/maddpg_mpe_config.yaml`.

## Running Training

To start training, run:

python scripts/train.py

text

## Testing

To evaluate the trained agents, run:

python scripts/test.py

text

## References

- Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)
- Multi-Agent Particle Environment (MPE) by OpenAI