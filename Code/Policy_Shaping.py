import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import math
import RL_functions as RL

class PSAgent:
    def __init__(self, action_space, state_space, gameName, participantID, alpha = 0.5, gamma=0.8, temp = 1, epsilon = 1):
        """
        Constructor method for the PSAgent class. Initializes the agent with the given parameters.
        """
        self.action_space = action_space * 2
        self.state_space = state_space
        self.alpha = alpha
        self.gamma= gamma
        self.temp = temp
        self.epsilon = epsilon
        self.gameName = gameName
        self.participantID = participantID
        self.rl = RL.RL(self.action_space, self.state_space)
        
        try:
            self.feedback = pd.read_csv('./userdata/'+self.gameName+'/'+self.participantID+'/'+self.gameName+'_PStable.csv', index_col=0)
            self.feedback.index = self.feedback.index.astype(str)  # Convert the index as str
            self.feedback.columns = self.feedback.columns.astype(int)
        except OSError as e:
            # Define the coordinates for the feedback matrix. USED ONLY IN THE PRE-TRAINING PHASE OF THE PROJECT
            self.feedback = self.rl.create_table()
            print(f"{e}\nCreating new file")
            
    def learning(self, action, feedback, state):
        """
        Update the feedback matrix with the given feedback.
        """
        state = self.rl.get_state(state)
        self.feedback.loc[state,action] += feedback

    def action_prob(self, state):
        """
        Calculate the probability of taking each action given the current state and feedback.
        """
        prob = []

        state = self.rl.get_state(state)
        
        if all(self.feedback.loc[state,:].to_numpy() == 0):
            return np.array([1/self.action_space for i in range(1, self.action_space+1)])
        for i in range(1, self.action_space+1):
            if self.feedback.loc[state, i] < -50:
                self.feedback.loc[state, i] = -50
            prob.append(math.pow(0.95,self.feedback.loc[state,i])/
                        (math.pow(0.95,self.feedback.loc[state,i]) +
                         math.pow(0.05,self.feedback.loc[state,i])) )
        return prob


if __name__ == "__main__":
    pass