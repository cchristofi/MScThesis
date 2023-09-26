import pandas as pd

class RL:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space

    def get_state(self, state):
        """
        Given a state, return the corresponding state in a specific format.
        """
        # If the state is not in the expected format, transform it to the expected format
        df = pd.DataFrame(state)

        if self.state_space == 6 or self.state_space == 7:
            # 9 CUTOFF POINTS
            cutoff = [-10.0, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 10.0]
            labels = ['invalid1', '2', '3', '4', '5', '6', '7', '8', 'invalid2']
        elif self.state_space == 3:
            # 16 CUTOFF POINTS
            cutoff = [-10.0, -0.64, -0.55, -0.46, -0.37, -0.28, -0.19, -0.1, -0.01, 0.08, 0.17, 0.26, 0.35, 0.44, 0.53, 0.62, 0.71, 0.8, 10.0]
            labels = ['invalid1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', 'invalid2']

        # 27 CUTOFF POINTS
        # cutoff = [-10.0, -0.7, -0.64, -0.58, -0.52, -0.46, -0.4, -0.34, -0.28, -0.22, -0.16, -0.1, -0.04, 0.02, 0.08, 0.14, 0.2, 0.26, 0.32, 0.38, 0.44, 0.5, 0.56, 0.62, 0.68, 0.74, 0.8, 10.0]
        # labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']
        
        df= pd.cut(df[0], bins=cutoff, labels=labels)
        state = df.tolist()
        state = "".join(state)
        return state
    
    def get_state_comma(self, state):
        """
        Given a state, return the corresponding state in a specific format.
        """
        # If the state is not in the expected format, transform it to the expected format
        df = pd.DataFrame(state)

        if self.state_space == 6 or self.state_space == 7:
            # 9 CUTOFF POINTS
            cutoff = [-10.0, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 10.0]
            labels = ['invalid1', '2', '3', '4', '5', '6', '7', '8', 'invalid2']
        elif self.state_space == 3:
            # 16 CUTOFF POINTS
            cutoff = [-10.0, -0.64, -0.55, -0.46, -0.37, -0.28, -0.19, -0.1, -0.01, 0.08, 0.17, 0.26, 0.35, 0.44, 0.53, 0.62, 0.71, 0.8, 10.0]
            labels = ['invalid1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', 'invalid2']

        # 27 CUTOFF POINTS
        # cutoff = [-10.0, -0.7, -0.64, -0.58, -0.52, -0.46, -0.4, -0.34, -0.28, -0.22, -0.16, -0.1, -0.04, 0.02, 0.08, 0.14, 0.2, 0.26, 0.32, 0.38, 0.44, 0.5, 0.56, 0.62, 0.68, 0.74, 0.8, 10.0]
        # labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27']
        
        df= pd.cut(df[0], bins=cutoff, labels=labels)
        state = df.tolist()
        state = ",".join(state)
        return state
    
    def create_table(self):
        """
        Creating the table with as many columns as the passed state space has which indicates the xyz position of the end-effector
        + the object if there is one + the width of the grip if it is available to open in the environment.
        list(range(1, 6)) -> 5 different states after being discretized.
        USED ONLY IN THE PRE-TRAINING PHASE OF THE PROJECT
        """
        if self.state_space == 3:
            x1 = list(range(2, 18)) # 2, 9
            y1 = list(range(2, 18)) # 2, 9
            z1 = list(range(2, 18)) # 2, 9
            self.coordinates = [f"{a1}{b1}{c1}" for a1 in x1 for b1 in y1 for c1 in z1]
            self.table = pd.DataFrame(index= self.coordinates, columns= range(1, self.action_space+1), dtype= object)
            self.table = self.table.fillna(1)

        elif self.state_space == 6:
            x1 = list(range(2, 9))
            y1 = list(range(2, 9))
            z1 = list(range(2, 9))
            x2 = list(range(2, 9))
            y2 = list(range(2, 9))
            z2 = list(range(2, 9))
            self.coordinates = [f"{a1}{b1}{c1}{a2}{b2}{c2}" for a1 in x1 for b1 in y1 for c1 in z1 for a2 in x2 for b2 in y2 for c2 in z2]
            self.table = pd.DataFrame(index= self.coordinates, columns= range(1, self.action_space+1), dtype= object)
            self.table = self.table.fillna(1)

        elif self.state_space == 7:
            x1 = list(range(2, 9))
            y1 = list(range(2, 9))
            z1 = list(range(2, 9))
            x2 = list(range(2, 9))
            x3 = list(range(2, 9))
            y3 = list(range(2, 9))
            z3 = list(range(2, 9))
            self.coordinates = [f"{a1}{b1}{c1}{a2}{a3}{b3}{c3}" for a1 in x1 for b1 in y1 for c1 in z1 for a2 in x2 for a3 in x3 for b3 in y3 for c3 in z3]
            self.table = pd.DataFrame(index= self.coordinates, columns= range(1, self.action_space+1), dtype= object)
            self.table = self.table.fillna(1)

        return self.table