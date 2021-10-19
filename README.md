# Reinforcement_Learning

## Reinforcement Learning Coursework on Deep Q-Learning at Imperial College London
Implemented a deep Q-network to solve randomly generated mazes. The implementation is divided in three files: random_environment.py, train_and_test.py, and agent.py
* random_environment.py creates random mazes - the agent always starts on the left of the maze and the goal is always on the right
* train_and_test.py gets an environment from random_environment.py, trains the agent for 10 minutes, and tests the agent's optimal policy on the maze
* Features implemented in the agent.py file:
  * Double Q-Learning with target network
  * Prioritized Experience Replay buffer
  * Early stopping
  * Epsilon-greedy policy
