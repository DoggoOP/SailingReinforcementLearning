# SailingReinforcementLearning
This repository contains a Reinforcement Learning project focused on simulating and optimizing sailing strategies using environmental data and physics-based simulations. The project utilizes data from computational studies discussed in a detailed computational essay available at this [link](https://community.wolfram.com/groups/-/m/t/2963984). Readers are highly encouraged to review the essay to understand the derivation of lift values and the decision-making process behind the actions within the simulated environment.

# The Environment
The SailingEnv class, defined in sailing_env.py, encapsulates the sailing simulation environment. It integrates realistic sailing dynamics, including wind effects, boat lift calculations, and penalty handling for suboptimal movements. The environment is built on top of the OpenAI Gym interface, making it compatible with various reinforcement learning algorithms provided by libraries like Stable Baselines3.

Key Features:
Dynamic Wind Field: Simulates varying wind conditions that affect the sailing strategy.
Lift Calculations: Uses predefined lift coefficients for different sail settings and angles relative to the wind.
Penalty System: Implements penalties for inefficient tacking and sailing out of bounds, encouraging the agent to learn optimal navigation paths.
The Training
Training processes are implemented using the Stable Baselines3 library, specifically the Proximal Policy Optimization (PPO) algorithm, which is well-suited for environments with continuous action spaces. The training script train_model.py manages the setup, execution, and saving of the RL model.

# Training Workflow:
Environment Setup: Instantiation of the SailingEnv with appropriate parameters.
Model Configuration: Configuration of the PPO model with specified hyperparameters.
Training Loop: The model trains over several episodes, with periodic evaluations to assess performance and improvements.
Model Saving: Post-training, the model is saved for future use and analysis.
The Visualization
The visualization.py script provides a graphical representation of the agent's behavior in the environment. It utilizes matplotlib to plot the trajectory of the sailboat as it navigates through the wind field, displaying real-time decision-making and movement adjustments.

# Visualization Features:
Trajectory Plotting: Shows the path taken by the sailboat from start to goal.
Wind Vectors: Depicts the wind conditions at various points in the environment, illustrating how wind affects movement decisions.
Action Highlights: Differentiates between sailing actions, such as tacking, to provide insights into strategy shifts.

# Getting Started
To get started with this project, clone the repository and ensure you have the necessary dependencies installed, including gym, numpy, and stable-baselines3. Run the train_model.py to train your model, followed by visualization.py to see the trained agent in action.


