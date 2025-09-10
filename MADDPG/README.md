# Project: Collaboration and Competition (Udacity DRL Nanodegree)

## Introduction

This project involves training a pair of agents to play a game of tennis. In this environment, two agents control rackets to bounce a ball over a net. The goal is to train the agents to collaborate by keeping the ball in play for as long as possible.

This is a multi-agent reinforcement learning task, solved using the **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm.

<img src="https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png"/>

*Agents playing tennis.*

---

## Project Environment Details

The simulation environment is provided by Unity Technologies.

*   **State Space:** The state space has **24 dimensions** per agent, corresponding to the position and velocity of the ball and racket. This is composed of 8 observations stacked 3 times.
*   **Action Space:** The action space is continuous, with **2 dimensions** per agent, corresponding to movement toward (or away from) the net and jumping. The values are in the range `[-1, 1]`.
*   **Reward Function:**
    *   If an agent hits the ball over the net, it receives a reward of **+0.1**.
    *   If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of **-0.01**.
*   **Solved Condition:** The environment is considered solved when the agents achieve an average score of **+0.5 over 100 consecutive episodes**, considering the maximum score of the two agents in each episode.

---

## Getting Started

### Prerequisites

*   Python 3.6
*   PyTorch
*   NumPy
*   Unity ML-Agents
*   Matplotlib
*   Tqdm

### Installation

1.  **Clone this repository and the Udacity DRL repository:**
    ```bash
    git clone https://github.com/your_username/drlnd-collaboration-competition.git
    cd drlnd-collaboration-competition
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    ```

2.  **Create and activate a new Python 3.6 environment.** Using Conda is recommended.
    ```bash
    conda create --name drlnd python=3.6
    conda activate drlnd
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip uninstall -y mlagents mlagents-envs
    cd deep-reinforcement-learning/python && pip install .
    ```

4.  **Download the Unity Environment:**
    Download the Tennis environment for your operating system,

---

## Instructions

To train the agents and see the results, follow these steps:

1.  Open the Jupyter Notebook `Report.ipynb`:
    ```bash
    jupyter notebook Report.ipynb
    ```
2.  Before running the code, ensure that the `file_name` variable in the notebook points to the correct location of the downloaded Tennis environment file.
3.  Execute the cells in the notebook sequentially to:
    *   Start the environment.
    *   Train the MADDPG agents. The weights of the trained actor and critic networks will be saved for each agent (e.g., `checkpoint_actor_0.pth`).
    *   View a plot of the rewards per episode.
    *   (Optional) Run the inference cells at the end to watch the trained agents perform.