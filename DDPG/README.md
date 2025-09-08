# Project: Continuous Control for Udacity DRL Nanodegree

## Introduction

This project involves training a reinforcement learning agent to control a double-jointed arm to reach target locations. The agent must learn to move the arm to a specific goal position in a continuous state and action space. This project uses the Deep Deterministic Policy Gradient (DDPG) algorithm to solve the environment.

<img src="https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif"/>

*Multi-agent Environment example*

---

## Project Environment Details

The simulation environment is provided by Unity Technologies and is based on the "Reacher" task.

*   **State Space:** The state space has **33 dimensions**, which include the position, rotation, velocity, and angular velocities of the two arm segments.
*   **Action Space:** The action space is continuous, with **4 dimensions**. Each dimension corresponds to the torque applied to the two joints, with values ranging from -1 to 1.
*   **Reward Function:** A reward of +0.1 is given for each simulation step that the agent's hand is within the goal location.
*   **Solved Condition:** The environment is considered solved when the agent achieves an average score of **+30 over 100 consecutive episodes**.

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

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/vlajnaya-mol/DeepRL.git
    cd drlnd-continuous-control
    ```
    
2. **Clone Udacity DRL repo [udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning/tree/master)**
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    ```

3.  **Create and activate a new Python 3.6 environment.** Using Conda is recommended.
    ```bash
    conda create --name drlnd python=3.6
    conda activate drlnd
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    pip uninstall -y mlagents mlagents-envs
    cd deep-reinforcement-learning/python && pip install . && cd ../..
    ```

5.  **Download the Unity Environment:**
    Download the Reacher environment for your operating system.

---

## Instructions

To train the agent and see the results, follow these steps:

1.  Open the Jupyter Notebook `Report.ipynb`:
    ```bash
    jupyter notebook Report.ipynb
    ```
2.  Before running the code, ensure that the `file_name` variable in the notebook points to the correct location of the downloaded Reacher environment file.
3.  Execute the cells in the notebook sequentially to:
    *   Start the environment.
    *   Train the DDPG agent. The weights of the trained actor and critic networks will be saved as `checkpoint_actor.pth` and `checkpoint_critic.pth` respectively.
    *   View a plot of the rewards per episode.
    *   (Optional) Run the inference cells at the end to watch the trained agent perform.