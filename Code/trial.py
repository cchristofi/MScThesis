import numpy as np
import shortuuid, time, yaml, logging
import pickle as cPickle
import gymnasium as gym
import rlbench.gym
import panda_gym
from agent import Agent
from Custom_env.CustomReachEnv import CustomReachEnv1, CustomReachEnv2, CustomReachEnv3
from datetime import timedelta, datetime
from timeit import default_timer as timer



# PUSH -------------------------------------------------------------------------------------------------------
instructions_intro_push = """<!DOCTYPE html><html lang="en"><body>
                        <p>Your goal is to teach the agent (robotic arm) to push the green box to the end goal (shaded box), no matter where the agent is placed in the environment. First, we ask you to click the keys on the keyboard to get familiar with the agent.</p>
                        <p>Once you have mastered the game, click <img src=":/Icons/icons/next20.png"/> on the top-right corner of the window and we will introduce you to the next step.</p>
                        </body></html>"""
# PUSH -------------------------------------------------------------------------------------------------------

# PICK AND PLACE -------------------------------------------------------------------------------------------------------
instructions_intro_pickandplace = """<!DOCTYPE html><html lang="en"><body>
                        <p>Your goal is to teach the agent (robotic arm) to pick up the green box and place is to the end goal (shaded box), no matter where the agent is placed in the environment. First, we ask you to click the keys on the keyboard to get familiar with the agent.</p>
                        <p>Once you have mastered the game, click <img src=":/Icons/icons/next20.png"/> on the top-right corner of the window and we will introduce you to the next step.</p>
                        </body></html>"""
# PICK AND PLACE -------------------------------------------------------------------------------------------------------

# REACH -------------------------------------------------------------------------------------------------------
instructions_intro_reach_task1 = """<!DOCTYPE html><html lang="en"><body>
                        <p>Your goal is to teach the agent (robotic arm) to reach to the end goal (shaded GREEN sphere), no matter where the agent is placed in the environment. First, we ask you to click the keys on the keyboard to get familiar with the agent.</p>
                        <p>Once you have mastered the game, click <img src=":/Icons/icons/next20.png"/> on the top-right corner of the window and we will introduce you to the next step.</p>
                        </body></html>"""

instructions_intro_reach_task2 = """<!DOCTYPE html><html lang="en"><body>
                        <p>Your goal is to teach the agent (robotic arm) to reach to the end goal (shaded GREEN sphere) by trying to AVOID the RED one, no matter where the agent is placed in the environment. First, we ask you to click the keys on the keyboard to get familiar with the agent.</p>
                        <p>Once you have mastered the game, click <img src=":/Icons/icons/next20.png"/> on the top-right corner of the window and we will introduce you to the next step.</p>
                        </body></html>"""

instructions_intro_reach_task3 = """<!DOCTYPE html><html lang="en"><body>
                        <p>Your goal is to teach the agent (robotic arm) to reach to the end goal (shaded GREEN sphere) by trying to TOUCH first the BLUE one, no matter where the agent is placed in the environment. First, we ask you to click the keys on the keyboard to get familiar with the agent.</p>
                        <p>Once you have mastered the game, click <img src=":/Icons/icons/next20.png"/> on the top-right corner of the window and we will introduce you to the next step.</p>
                        </body></html>"""
# REACH -------------------------------------------------------------------------------------------------------

instructions_position = """<!DOCTYPE html><html lang="en"><body>
                            <p>Congratulations on mastering the control of the agent. Now we will introduce you to the different options you can use to teach the agent how to successfully complete the task by itself. First we start with selecting the starting state position. During the real game, think about how to strategically place the agent according to what it already knows so that it can best learn.</p>
                            <p>You can select the starting position of the agent by using the keys on the keyboard that are shown on the screen. When you have selected the starting position in the real game, click <img src=":/Icons/icons/play20.png"/> to start the learning process.</p>
                            </body></html>"""

instructions_demo = """<!DOCTYPE html><html lang="en"><body>
                        <p>You can demonstrate to the agent what it should do by taking control and using the keys to show it the path! You decide how long the demonstration should be it could be as short as a single move, or as long as a full path to the goal.</p>
                        <p>To enter demonstration mode, click <img src=":/Icons/icons/pause20.png"/> and press the keys shown on your screen to give your input. Once you are done, press <img src=":/Icons/icons/play20.png"/> to go back to the normal game speed. You can enter into demonstration mode any time you like, but there will be a limited quota for how total many moves you can demonstrate throughout the game (shown on the bottom of the window). So use it wisely!</p>
                        <p>Take a few moments to practice this is not the real game just yet. If you are ready to proceed, click <img src=":/Icons/icons/next20.png"/>.</p>
                        </body></html>"""

instructions_feedback = """<!DOCTYPE html><html lang="en"><body>
                            <p>You will see two buttons on the bottom right corner of the window, <img src=":/Icons/icons/feedbackGood20.png"/> and <img src=":/Icons/icons/feedbackBad20.png"/>. You can click on either of these to train the agent by giving it feedback on its last FIVE moves, not its general behaviour!</p>
                            <p>There will be a limited quota for how many times you can give the feedback (shown on the bottom of the window). So use it wisely! If you are ready to start, click <img src=":/Icons/icons/next20.png"/>.</p>
                            </body></html>"""

instructions_final = """<!DOCTYPE html><html lang="en"><body>
                        <p>Well done!! You are now ready to train the agent. You have a limited quote for demonstrations and feedback to train the agent. Important to remember is that the final teaching performance is based on how well the agent learns to complete the task, no matter where it is randomly placed on the grid.</p>
                        <p>Click <img src=":/Icons/icons/next20.png"/> to select the first starting point.</p>
                        </body></html>"""

instructions_game = """<!DOCTYPE html><html lang="en"><body>
                <p>After selecting the starting point of the agent, click <img src=":/Icons/icons/play20.png"/> to start the simulation. To take control of the agent, click <img src=":/Icons/icons/pause20.png"/> and click the keys that you see below to move the agent where you want.</p>
                <p>To reset the whole trial click <img src=":/Icons/icons/reset20.png"/> and select a NEW starting position.</p>
                <p>When the agent is performing by itself, you can give "Good" / "Bad" feedback by clicking <img src=":/Icons/icons/feedbackGood20.png"/> / <img src=":/Icons/icons/feedbackBad20.png"/> respectively.</p>
                <p>If you want to end the experiment, click <img src=":/Icons/icons/stop20.png"/>.</p>
                </body></html>"""

instructions_end = """<!DOCTYPE html><html lang="en"><body>
                <p>Congratulations, you have successfully taught the agent to complete the task.</p>
                <p>To end the experiment, click <img src=":/Icons/icons/stop20.png"/> or press <img src=":/Icons/icons/key_esc.png"/>.</p>
                </body></html>"""

def load_config():
    logging.info('Loading Config in trial.py')
    with open(r'.trialConfig.yml', 'r') as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    logging.info('Config loaded in trial.py')
    return config.get('trial')

class Trial():
    def __init__(self, alterText, runTrial, participantID, taskName):
        self.init_variables()
        # Signals
        self.alterText = alterText
        self.runTrial = runTrial
        self.participant_ID = participantID
        self.taskName = taskName
        self.runTrial.connect(self.start)

    def init_variables(self):
        '''
        Initialization of all the parameters.
        '''
        self.config = load_config()
        self.frameId = 0
        self.humanAction = 0
        self.episode = 0
        self.play = False
        self.record = []
        self.nextEntry = {}
        self.trialId = shortuuid.uuid()
        self.outfile = None
        self.userId = None
        self.projectId = self.config.get('projectId')
        self.filename = None
        self.path = None
        self.humanFeedback = 'none'
        self.start_time = time.time()
        self.render = {}
        self.message = {}
        self.saved_state_id = None
        self.trainingPhase = True
        self.phase = 0
        ################# EXPERIMENT INFORMATION ############
        # self.participant_ID = '0' # Retrieving from GUI
        self.gameName = 'PandaReach-v3' # PandaReachDense-v3 # PandaPushDense-v3 # PandaPickAndPlaceDense-v3
        # self.taskName = 'CustomReachEnv1' # CustomReachEnv1 # CustomReachEnv2 # CustomReachEnv3 # Retrieving from GUI
        ############################################

    def start(self):
        '''
        Call the function in the Agent/Environment combo required to start 
        a trial. By default passes the environment name that will be passed
        to gym.make(). 
        By default this expects the openAI Gym Environment object to be
        returned. 
        '''
        
        if self.taskName == 'CustomReachEnv1' or self.taskName == 'CustomReachEnv0':
            self.env = CustomReachEnv1(render_mode="human",
                        render_target_position=[-0.2, 0, 0],
                        render_distance=1.0,
                        render_yaw=90,
                        render_pitch=-50,
                        render_roll=0)
        elif self.taskName == 'CustomReachEnv2':
            self.env = CustomReachEnv2(render_mode="human",
                        render_target_position=[-0.2, 0, 0],
                        render_distance=1.0,
                        render_yaw=90,
                        render_pitch=-50,
                        render_roll=0)
        elif self.taskName == 'CustomReachEnv3':
            self.env = CustomReachEnv3(render_mode="human",
                        render_target_position=[-0.2, 0, 0],
                        render_distance=1.0,
                        render_yaw=90,
                        render_pitch=-50,
                        render_roll=0)

        self.agent = Agent()
        self.agent.start(self.env, self.gameName, self.participant_ID, self.taskName)
        self.run()

    def run(self):
        '''
        This is the main event controlling function for a Trial. 
        It handles the render-step loop
        '''
        self.reset()
        self.send_render()
        while not self.agent.end: 
        
            if self.message:
                self.handle_message(self.message)

            if self.play or self.agent.demo:
                self.get_render()
                self.send_render()
                self.take_step()
        
        self.end()

    def reset(self):
        '''
        Resets the OpenAI gym environment to start a new episode.
        By default this function will create a new log file for every
        episode, if the intention is to log only full trials then
        comment the 3 lines below contianing self.outfile and 
        self.create_file.
        '''
        if self.check_trial_done():
            self.end()
        else:
            self.agent.reset()
            if self.outfile:
                self.outfile.close()
            self.create_file()
            self.episode += 1

    def check_trial_done(self):
        '''
        Checks if the trial has been completed and can be quit. Add conditions
        as required.
        '''
        return self.episode >= 1
    # self.config.get('maxEpisodes', 20)
    
    def end(self):
        '''
        Closes the environment through the agent, closes any remaining outfile
        and sends the 'done' message to the websocket pipe. If logging the 
        whole trial memory in self.record, uncomment the call to self.save_record()
        to write the record to file before closing.
        '''
        print('done')
        self.agent.close()
        # if self.config.get('dataFile') == 'trial':
        #     self.save_record()
        if self.outfile:
            self.outfile.close()
            print({'upload':{'projectId':self.projectId,'participantID':self.participant_ID,'file':self.filename,'path':self.path}})
        self.play = False
        self.agent.end = True
        self.send_render()

    def handle_message(self, message:dict):
        '''
        Reads messages sent from websocket, handles commands as priority then 
        actions. Logs entire message in self.nextEntry
        '''
        if 'command' in message and message['command']:
            self.handle_command(message['command'])
            message['command'] = 'None'
        if 'feedback' in message and message['feedback']:
            self.handle_feedback(message['feedback'])
            message['feedback'] = 'none'
        if 'action' in message and message['action']:
            self.handle_action(message['action'])
            message['action'] = 'none'
        self.update_entry(message)

    def handle_command(self, command:str='None'):
        '''
        Deals with allowable commands from user. To add other functionality
        add commands.
        '''
        command = command.strip().lower()
        if command == 'start':
            if self.agent.elaps_time.time == 0.0:
                self.agent.elaps_time.start()
            self.agent.demo = False
            self.play = True
        elif command == 'stop':
            self.agent.end = True
            return
        elif command == 'reset':
            self.play = False
            self.agent.demo = False
            self.agent.reset()
        elif command == 'pause':
            self.play = False
            self.agent.demo = True
            
    def handle_action(self, action:str):
        '''
        Translates action to int and resets action buffer if action !=0
        '''
        action = action.strip().lower()

        actionSpace = self.config.get('actionSpace')
        if action in actionSpace:
            actionCode = actionSpace.index(action) + 1
        else:
            actionCode = 0
        
        self.humanAction = actionCode

    def handle_feedback(self, feedback:str):
        '''
        Translates feedback
        '''
        self.humanFeedback = feedback

    def update_entry(self, update_dict:dict):
        '''
        Adds a generic dictionary to the self.nextEntry dictionary.
        '''
        self.nextEntry.update(update_dict)

    def get_render(self):
        '''
        Calls the Agent/Environment render function which must return a npArray.
        Translates the npArray into a jpeg image and then base64 encodes the 
        image for transmission in json message.
        '''
        render = self.agent.render()
    
    def send_render(self):
        '''
        Sends render dictionary to GUI in order to show the required information to the user.
        '''
        if not self.agent.end:
            if self.phase == 0: # Intro Instructions
                if self.gameName == "PandaPushDense-v3":
                    self.render['display'] = {'Game Instructions': instructions_intro_push,
                                        'Demos Left': self.agent.demo_steps,
                                        'Feedback Left': self.agent.feedback_steps}
                elif self.gameName == "PandaPickAndPlaceDense-v3":
                    self.render['display'] = {'Game Instructions': instructions_intro_pickandplace,
                                        'Demos Left': self.agent.demo_steps,
                                        'Feedback Left': self.agent.feedback_steps}
                elif self.gameName == "PandaReachDense-v3"  or self.gameName == "PandaReach-v3":
                    if self.taskName == "CustomReachEnv1" or self.taskName == "CustomReachEnv0":
                        self.render['display'] = {'Game Instructions': instructions_intro_reach_task1,
                                            'Demos Left': self.agent.demo_steps,
                                            'Feedback Left': self.agent.feedback_steps}
                    elif self.taskName == "CustomReachEnv2":
                        self.render['display'] = {'Game Instructions': instructions_intro_reach_task2,
                                            'Demos Left': self.agent.demo_steps,
                                            'Feedback Left': self.agent.feedback_steps}
                    elif self.taskName == "CustomReachEnv3":
                        self.render['display'] = {'Game Instructions': instructions_intro_reach_task3,
                                            'Demos Left': self.agent.demo_steps,
                                            'Feedback Left': self.agent.feedback_steps}
            elif self.phase == 1: # Position Instructions
                self.render['display'] = {'Game Instructions': instructions_position,
                                    'Demos Left': self.agent.demo_steps,
                                    'Feedback Left': self.agent.feedback_steps}
            elif self.phase == 2: # Demo Instructions
                self.render['display'] = {'Game Instructions': instructions_demo,
                                    'Demos Left': self.agent.demo_steps,
                                    'Feedback Left': self.agent.feedback_steps}
            elif self.phase == 3: # Feedback Instructions
                self.render['display'] = {'Game Instructions': instructions_feedback,
                                    'Demos Left': self.agent.demo_steps,
                                    'Feedback Left': self.agent.feedback_steps}
            elif self.phase == 4: # Final Instructions
                self.render['display'] = {'Game Instructions': instructions_final,
                                    'Demos Left': self.agent.demo_steps,
                                    'Feedback Left': self.agent.feedback_steps}
            else: # Game Instructions
                self.render['display'] = {'Game Instructions': instructions_game,
                                        'Demos Left': self.agent.demo_steps,
                                        'Feedback Left': self.agent.feedback_steps}
        else:
            self.render['display'] = {'Game Instructions': instructions_end,
                                        'Demos Left': 0,
                                        'Feedback Left': 0}
            
        self.alterText.emit(self.render)

    def take_step(self):
        '''
        Expects a dictionary return with all the values that should be recorded.
        Records return and saves all memory associated with this setp.
        Checks for DONE from Agent/Env
        '''
        envState = self.agent.step(self.humanAction, self.humanFeedback)
        if self.agent.invalidState or self.agent.invalidNextState:
            self.play = False
            self.agent.demo = False
            self.send_render()
            return
        self.update_entry(envState)
        self.save_entry()
        self.humanFeedback = 'none'
        self.humanAction = 0
        self.send_render()

        if self.agent.demo_steps <= 0 and self.agent.feedback_steps <=0:
            self.agent.end = True
            self.play = False
            self.agent.demo = False
            return

        if self.agent.trial_timer >= 180:
            self.agent.end = True
            self.play = False
            self.agent.demo = False
            return

        if envState['done']:
            self.play = False
            self.agent.demo = False
            self.agent.reset()
            return

    def save_entry(self):
        '''
        Either saves step memory to self.record list or pickles the memory and
        writes it to file, or both.
        Note that observation and render objects can get large, an episode can
        have several thousand steps, holding all the steps for an episode in 
        memory can cause performance issues if the os needs to grow the heap.
        The program can also crash if the Server runs out of memory. 
        It is recommended to write each step to file and not maintain it in
        memory if the full observation is being saved.
        comment/uncomment the below lines as desired.
        '''
        if self.config.get('dataFile') == 'trial':
            self.record.append(self.nextEntry)
        else:
            cPickle.dump(self.nextEntry, self.outfile)
            self.nextEntry = {}

    def save_record(self):
        '''
        Saves the self.record object to file. Is only called if uncommented in
        self.end(). To record full trial records a line must also be uncommented
        in self.save_entry() and self.create_file()
        '''
        cPickle.dump(self.record, self.outfile)
        self.record = []

    def create_file(self):
        '''
        Creates a file to record records to. comment/uncomment as desired 
        for episode or full-trial logging.
        '''
        if self.config.get('dataFile') == 'trial':
            filename = f'trial_{self.participant_ID}'
        else:
            filename = f'episode_{self.episode}_participant_{self.participant_ID}'
        path = './Trials/'+filename
        self.outfile = open(path, 'ab')
        self.filename = filename
        self.path = path
        

if __name__ == "__main__":
    pass
