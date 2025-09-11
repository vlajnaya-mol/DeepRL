# DeepRL

Implementations of core Deep Reinforcement Learning (DRL) algorithms in PyTorch for Udacity Deep Reinforcement Learning course.

## Algorithms
- **DQN** — value-based learning for discrete actions  
- **DDPG** — actor–critic for continuous actions  
- **MADDPG** — multi-agent extension of DDPG

## Project structure
```

DeepRL/
├─ DQN/         # Deep Q-Network implementations & notebooks
├─ DDPG/        # Deep Deterministic Policy Gradient
├─ MADDPG/      # Multi-Agent DDPG
├─ utils.py     # shared helpers/utilities
├─ **init**.py
└─ .gitignore

````

## Notes
* Most experiments are notebook-driven; check each folder for environment choice, hyperparameters, and logging.
* Check each algorithm's README.md file for the env setup guidelines.
* Open the notebooks in the algorithm folder you’re interested in (e.g., `DQN/`, `DDPG/`, `MADDPG/`) and run the cells.
* Issues and PRs for improvements, new environments, and benchmarks are welcome.
