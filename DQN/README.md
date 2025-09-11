# Project 1: Navigation (Udacity DRL Nanodegree)

## Introduction

This project involves training a reinforcement learning agent to navigate a large, square world and collect bananas. The goal is to train an agent to collect as many yellow bananas as possible while avoiding blue bananas.

This task is solved using a **Deep Q-Network (DQN)** agent, enhanced with **Prioritized Experience Replay (PER)** to improve learning efficiency.

<img src="https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif"/>

*Trained agent collecting bananas.*

---

## Project Environment Details

The simulation environment is provided by Unity Technologies.

*   **State Space:** The state space has **37 dimensions**, which include the agent's velocity and ray-based perception of objects in its forward direction.
*   **Action Space:** The action space is discrete, with **4 possible actions**:
    *   `0`: Move forward
    *   `1`: Move backward
    *   `2`: Turn left
    *   `3`: Turn right
*   **Reward Function:**
    *   **+1** for collecting a yellow banana.
    *   **-1** for collecting a blue banana.
*   **Solved Condition:** The environment is considered solved when the agent achieves an average score of **+13 over 100 consecutive episodes**.

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
    git clone https://github.com/your_username/drlnd-navigation.git
    cd drlnd-navigation
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
    Download the Banana environment for your operating system.

---

## Instructions

To train the agent and see the results, follow these steps:

1.  Open the Jupyter Notebook `Report.ipynb`:
    ```bash
    jupyter notebook Report.ipynb
    ```
2.  Before running the code, ensure that the `file_name` variable in the notebook points to the correct location of the downloaded Banana environment file.
3.  Execute the cells in the notebook sequentially to:
    *   Start the environment.
    *   Train the DQN agent. The weights of the trained Q-network will be saved as `checkpoint_qn_local.pth`.
    *   View a plot of the rewards per episode.
    *   (Optional) Run the inference cells at the end to watch the trained agent perform.