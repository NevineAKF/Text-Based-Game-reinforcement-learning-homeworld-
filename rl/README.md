Q-Learning and Deep Q-Network (DQN) for Text-Based Reinforcement Learning

This repository contains a complete implementation of tabular Q-learning, linear function approximation, and Deep Q-Networks (DQN) for a deterministic, text-based reinforcement learning environment inspired by MIT’s Machine Learning 6.86 coursework.

The project demonstrates how different reinforcement learning approaches perform when the agent operates over textual state descriptions, which must be transformed into numerical representations before learning can occur.

🚀 Project Overview

This repository includes three progressive RL agents:

Tabular Q-Learning Agent
Learns the Q-function exactly using a lookup table.
Works for small, discrete state spaces.

Linear Approximation Q-Learning Agent
Uses a bag-of-words encoding combined with a linear model to approximate Q-values.
Reduces memory consumption and enables learning in larger state spaces.

Deep Q-Network (DQN) Agent
Implements a neural network using PyTorch to approximate Q(s, a).
Learns directly from text-based bag-of-words state vectors.

The environment presents the agent with:

A room description (text)

A quest description (text)

A set of actions and objects

The goal is to learn the correct sequence of (action, object) pairs to complete quests efficiently.

📂 Repository Structure
│── agent_tabular_ql.py     # Tabular Q-learning implementation
│── agent_linear.py         # Linear approximation Q-learning
│── agent_dqn.py            # Deep Q-Network (DQN) implementation
│── framework.py            # Environment logic and game engine
│── utils.py                # Bag-of-words, helper functions, ewma, indexing
│── game.tsv                # Game dataset (textual states)
│── README.md               # This documentation
└── requirements.txt        # Python dependencies

🔧 Key Features
✔ Text-to-State Encoding

Uses bag-of-words to convert textual room and quest descriptions into fixed-length vectors.

✔ Multiple RL Approaches

Tabular Q-learning

Linear Q-learning with feature engineering

Deep Q-learning using a neural network

✔ Epsilon-Greedy Exploration

Adjustable exploration rate for training/testing.

✔ Hyperparameter Control

Number of runs

Episodes per epoch

Discount factor γ

Learning rate α

Testing vs training ε

✔ Performance Monitoring

Tracks:

Average episodic reward

EWMA (exponentially weighted moving average)

Plots reward convergence over training epochs.

🧠 Deep Q-Network Architecture

The DQN processes bag-of-words state vectors and outputs independent Q-values for:

Each possible action

Each possible object

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)
        self.state2object = nn.Linear(hidden_size, object_dim)


The final Q-value for a command (action, object) is computed as:

Q(s, a, o) = 0.5 * (Q_action(a) + Q_object(o))

📊 Training Procedure

Each epoch consists of:

Training phase:

Run NUM_EPIS_TRAIN episodes

Update the model via tabular, linear, or deep Q-learning

Testing phase:

Run NUM_EPIS_TEST episodes

Compute cumulative discounted reward

Average results over multiple runs

The final output includes:

Convergence curves

Average reward per epoch

EWMA performance trends

▶️ How to Run
1. Install Dependencies
pip install -r requirements.txt

2. Run Tabular Q-Learning
python agent_tabular_ql.py

3. Run Linear Approximation
python agent_linear.py

4. Run Deep Q-Learning (DQN)
python agent_dqn.py

🧪 Experiments

This project allows evaluating:

Effect of ε on exploration vs exploitation

Impact of α on convergence behavior

Differences between tabular, linear, and deep Q-learning performance

Stability and reward convergence across epochs




📈 Results Summary
Tabular Q-learning converges quickly on small state spaces.
Linear approximation struggles to fully learn optimal policies, showing limitations of linear models for text-based RL.
DQN significantly improves performance and can approximate Q-values more effectively due to nonlinear representation learning.




🤝 Acknowledgment
This project is inspired by theoretical and practical frameworks introduced in:
MIT 6.86x — Machine Learning with Python
All environment logic and datasets follow the educational structure of the course.




📜 License
This repository is released under the MIT License.
