import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math
import RL_functions as RL

class QLAgent:
    def __init__(self, action_space, state_space, gameName, participantID, alpha = 0.5, gamma=0.8, temp = 1, epsilon = 0.95, mini_epsilon = 0.01, decay = 0.999):
        self.action_space = action_space * 2
        self.state_space = state_space
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.epsilon = epsilon
        self.mini_epsilon = mini_epsilon
        self.decay = decay
        self.gameName = gameName
        self.participantID = str(participantID)
        self.rl = RL.RL(self.action_space, self.state_space)

        try:
            self.qtable = pd.read_csv('./userdata/'+self.gameName+'/'+self.participantID+'/'+self.gameName+'_Qtable.csv', index_col=0)          
            self.qtable.index = self.qtable.index.astype(str)  # Convert the index as str
            self.qtable.columns = self.qtable.columns.astype(int)
        except OSError as e:
            # Define the coordinates for the q-table. USED ONLY IN THE PRE-TRAINING PHASE OF THE PROJECT
            self.qtable = self.rl.create_table()
            print(f"{e}\nCreating new file")

    def learning(self, action, rwd, state, next_state):
        state = self.rl.get_state(state)
        next_state = self.rl.get_state(next_state)
        q_sa = self.qtable.loc[state, action]
        max_next_q_sa = self.qtable.loc[next_state, :].max()
        new_q_sa = q_sa + self.alpha * (rwd + self.gamma * max_next_q_sa - q_sa)
        self.qtable.loc[state, action] = new_q_sa
        return new_q_sa, max_next_q_sa

    def action_prob(self, state):
        state = self.rl.get_state(state)
        p = np.random.uniform(0,1)

        if p <= self.epsilon:
            return np.array([1/self.action_space for i in range(1, self.action_space+1)])
        else:
            prob = F.softmax(torch.tensor(self.qtable.loc[state, :].to_list()).float(),dim = 0).detach().numpy()
            return prob

    def choose_action(self, state):
        state = self.rl.get_state(state)
        p = np.random.uniform(0,1)
        if self.epsilon >= self.mini_epsilon:
            self.epsilon *= self.decay
        if p <= self.epsilon:
            return np.random.choice([i for i in range(1, self.action_space+1)])
        else:
            action = np.argmax(self.qtable.loc[state, :]) + 1
            return action
        

if __name__ == "__main__":
    pass