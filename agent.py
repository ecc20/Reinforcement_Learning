############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
# Eric Chen (CID : 01936805)
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import collections


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 200
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
       
        # A dictionary of all possible actions in the environment 
        self.continuous_action = {0 : np.array([0.02, 0], dtype=np.float32), # Move 0.02 to the right
                                  1 : np.array([0, 0.02], dtype=np.float32), # Move 0.02 to the top
                                  2 : np.array([0, -0.02], dtype=np.float32)} # Move 0.02 to the bottom
        # The corresponding key in the dictionary for the action
        self.discrete_action = None
        self.replay_buffer = ReplayBuffer()
        self.minibatch_size = 200
        self.dqn = DQN()
        
        # The frequency for which we update the target network
        self.update_freq = 100
        
        # The epsilon for exploration
        self.epsilon = 1
        self.initial_epsilon = 1 
        self.epsilon_decay = 0.99
        
        # Early stopping
        self.early_stopping = False
        self.early_stopping_count = 0
        self.episode_count = 0
        
        # Prioritized Experience Replay
        self.constant = 0.001
        
    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.early_stopping_count = 0
            self.episode_count += 1 
            return True
        else:
            return False
    
    # Choose the continuous action 
    def _choose_next_action(self, state):
        # Implement the epsilon-greedy policy
        policy = np.ones(3)*self.epsilon/3
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_value = self.dqn.q_network.forward(state_tensor).detach().numpy()
        policy[np.argmax(q_value)] = 1 - self.epsilon + self.epsilon/3 
        
        # Continuous action
        self.discrete_action = np.random.choice(range(3),p=policy)
        continuous_action = self.continuous_action[self.discrete_action]
        return continuous_action
    
    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        
        # Early stopping counter
        self.early_stopping_count += 1
        if self.early_stopping_count <=100 and self.episode_count % 3 == 0:
            #print("Greedy policy from ", state)
            return self.get_greedy_action(state)
        if not self.early_stopping:
            #print("Executing epsilon-greedy policy")
            #print('ep',self.epsilon)
            action = self._choose_next_action(state)
        else:
            #print("Executing Greedy policy forever")
            action = self.get_greedy_action(state)
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        if not self.early_stopping:
            if self.early_stopping_count <= 100 and self.episode_count % 3 == 0:
                if distance_to_goal < 0.03:
                    self.early_stopping = True
                    print("Early Stopping")
                    return
            else:
                # Convert the distance to a reward
                reward = 0.1*(1 - distance_to_goal)**2
                
                # Create a transition (we use discrete_action instead of action)
                transition = (self.state, self.discrete_action, reward, next_state)
                self.replay_buffer.add_transition(transition)
                
                # Prioritized experience replay
                if self.num_steps_taken == 1: 
                    max_weight = self.constant # To have a weight at the very beginning of the training
                else:
                    max_weight = self.replay_buffer.get_max_weight()
                
                self.replay_buffer.add_weights(max_weight)
                
                if (len(self.replay_buffer.buffer) > self.minibatch_size):
                    mini_batch = self.replay_buffer.sample_minibatch(self.minibatch_size)
                    loss, weight = self.dqn.train_q_network(mini_batch)
                    self.replay_buffer.set_weights(weight)
                    
                    self.epsilon = (self.initial_epsilon*self.epsilon_decay**(self.episode_count-1))*(1 - (self.episode_length-(self.num_steps_taken%self.episode_length))/self.episode_length)
                    if (self.num_steps_taken % self.update_freq == 0):
                        self.dqn.update_target()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_value = self.dqn.q_network.forward(state_tensor).detach().numpy()
        discrete_action = np.argmax(q_value)
        action = self.continuous_action[discrete_action]
        # Returns the action with the highest Q-value
        return action
    
    
# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=3)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Create a target network
        self.target_network = Network(input_dimension=2, output_dimension=3)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Prioritized Experience Replay
        self.constant = 0.001
        self.alpha = 1
    
    # Update the target network
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        return self
    
    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss, weight = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item(), weight

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, mini_batch):
        
        # Double Q-Learning
        learning_rate = 0.9
        states, actions, rewards, next_states = zip(*mini_batch)
        states_tensor = torch.tensor(states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64) 
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        
        state_q_value_tensor = self.q_network.forward(states_tensor)
        prediction_tensor = state_q_value_tensor.gather(dim=1, index=actions_tensor.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            target_next_state_q_values_tensor = self.target_network.forward(next_states_tensor).detach()
            max_q_values, max_idx = torch.max(target_next_state_q_values_tensor, dim=1)
            next_state_q_values_tensor = self.q_network.forward(next_states_tensor).detach()
        
        # Bellman target
        target = rewards_tensor + learning_rate * next_state_q_values_tensor.gather(dim=1, index=max_idx.unsqueeze(-1)).squeeze(-1)
        loss = torch.nn.MSELoss()(prediction_tensor, target)
        
        # Compute weight for prioritized experience replay
        with torch.no_grad():
            delta = target - prediction_tensor
            weight = (np.abs(delta.numpy()) + self.constant*np.ones(len(mini_batch)))**self.alpha
        return loss, weight


class ReplayBuffer:
    
    # Class attribute
    buffer = collections.deque(maxlen=10000)
    weights = collections.deque(maxlen=10000)
    max_weight = 0 
    minibatch_indices = None
    
    def add_transition(self, transition):
        self.buffer.append(transition)
    
    def add_weights(self, weight):
        self.weights.append(weight)
        
    def set_weights(self, weight_list):
        w = np.array(self.weights)
        w[self.minibatch_indices] = weight_list
        self.weights = collections.deque(w, maxlen=10000)
    
    def get_max_weight(self):
        self.max_weight = max(self.weights)
        return self.max_weight
    
    def normalize(self):
        return np.array(self.weights)/np.sum(np.array(self.weights))
        
    def sample_minibatch(self, minibatch_size):
        self.minibatch_indices = np.random.choice(range(len(self.buffer)), minibatch_size, replace=False, p=self.normalize())
        mini_batch = [self.buffer[idx] for idx in self.minibatch_indices]
        return mini_batch
    