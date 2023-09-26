from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QFrame, QProgressBar
from PyQt5 import uic
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt
import sys
from trial import Trial
from .resources import *
import numpy as np


class RunTrial(QObject):
    start = pyqtSignal()
    runTrial = pyqtSignal()
    alterText = pyqtSignal(dict)
    def __init__(self, participantID, taskName):
        self.participantID = participantID
        self.taskName = taskName
        super().__init__()
        
    def run(self):
        self.trial = Trial(self.alterText, self.runTrial, self.participantID, self.taskName)
        self.start.emit()
        

class UI(QMainWindow):
    def __init__(self, participantID, taskName):
        self.participantID = participantID
        self.taskName = taskName
        super(UI, self).__init__()

        # Load the UI file
        uic.loadUi('./GUI/form.ui', self)

        # Define our widgets
        self.lblGameInstructions = self.findChild(QLabel, "lblGameInstructions")
        self.lblDemoInstructions = self.findChild(QLabel, "lblDemoInstructions")
        self.lblDemoInstructionsGrip = self.findChild(QLabel, "lblDemoInstructionsGrip")
        self.lblDemoInstructionsTitle = self.findChild(QLabel, "lblDemoInstructionsTitle")
        self.lblCustomTask = self.findChild(QLabel, "lblCustomTask")
        self.frameDemoInstructions = self.findChild(QFrame, "frameDemoInstructions")
        self.frameControls = self.findChild(QFrame, "frameControls")
        self.frameSteps_Feedback = self.findChild(QFrame, "frameSteps_Feedback")
        self.pbDemo = self.findChild(QProgressBar, "pbDemo")
        self.pbFeedback = self.findChild(QProgressBar, "pbFeedback")
        self.frameFeedback = self.findChild(QFrame, "frameFeedback")
        self.btnStart = self.findChild(QPushButton, "btnStart")
        self.btnPause = self.findChild(QPushButton, "btnPause")
        self.btnStop = self.findChild(QPushButton, "btnStop")
        self.btnReset = self.findChild(QPushButton, "btnReset")
        self.btnFeedbackGood = self.findChild(QPushButton, "btnFeedbackGood")
        self.btnFeedbackBad = self.findChild(QPushButton, "btnFeedbackBad")
        self.btnNext = self.findChild(QPushButton, "btnNext")

        self.lblGameInstructions.setWordWrap(True)
        self.lblDemoInstructions.setWordWrap(True)

        # Set functionality of control buttons
        self.btnStart.clicked.connect(self.controlBehaviour)
        self.btnPause.clicked.connect(self.controlBehaviour)
        self.btnStop.clicked.connect(self.controlBehaviour)
        self.btnReset.clicked.connect(self.controlBehaviour)

        # Set functionality of feedback buttons
        self.btnFeedbackGood.clicked.connect(self.getFeedback)
        self.btnFeedbackBad.clicked.connect(self.getFeedback)

        # Start Signals
        self.initSignals()

        # Show the app
        self.show()

        # Teacher training phase
        self.btnNext.clicked.connect(self.trainingPhase)

        self.frameFeedback.setEnabled(False)

        self.new_thread.start()

    def initSignals(self):
        self.runTrial = RunTrial(self.participantID, self.taskName)
        self.new_thread = QThread()
        self.runTrial.moveToThread(self.new_thread)
        self.new_thread.started.connect(self.runTrial.run)
        self.runTrial.start.connect(self.startSignals)

    def startSignals(self):
        self.runTrial.trial.alterText.connect(self.setLblText)
        self.runTrial.trial.runTrial.emit()
        self.getTrainingPhase()

    def closeEvent(self, event):
        self.btnStop.click()
        self.new_thread.quit()
        self.new_thread.wait()

    def keyPressEvent(self, qKeyEvent):
        if self.runTrial.trial.agent.demo:
            if self.runTrial.trial.gameName == 'PandaPickAndPlaceDense-v3':
                if qKeyEvent.key() == Qt.Key_W:
                    self.runTrial.trial.message['action'] = 'forward'
                if qKeyEvent.key() == Qt.Key_S:
                    self.runTrial.trial.message['action'] = 'backward'
                if qKeyEvent.key() == Qt.Key_A:
                    self.runTrial.trial.message['action'] = 'left'
                if qKeyEvent.key() == Qt.Key_D:
                    self.runTrial.trial.message['action'] = 'right'
                if qKeyEvent.key() == Qt.Key_8:
                    self.runTrial.trial.message['action'] = 'up'
                if qKeyEvent.key() == Qt.Key_2:
                    self.runTrial.trial.message['action'] = 'down'
                if qKeyEvent.key() == Qt.Key_4:
                    self.runTrial.trial.message['action'] = 'open'
                if qKeyEvent.key() == Qt.Key_6:
                    self.runTrial.trial.message['action'] = 'close'
            else:
                if qKeyEvent.key() == Qt.Key_W:
                    self.runTrial.trial.message['action'] = 'forward'
                if qKeyEvent.key() == Qt.Key_S:
                    self.runTrial.trial.message['action'] = 'backward'
                if qKeyEvent.key() == Qt.Key_A:
                    self.runTrial.trial.message['action'] = 'left'
                if qKeyEvent.key() == Qt.Key_D:
                    self.runTrial.trial.message['action'] = 'right'
                if qKeyEvent.key() == Qt.Key_8:
                    self.runTrial.trial.message['action'] = 'up'
                if qKeyEvent.key() == Qt.Key_2:
                    self.runTrial.trial.message['action'] = 'down'

        if self.runTrial.trial.trainingPhase or self.runTrial.trial.agent.startingPoint:
            if self.runTrial.trial.gameName == 'PandaPickAndPlaceDense-v3':
                if qKeyEvent.key() == Qt.Key_W:
                    self.runTrial.trial.agent.take_training_step([1, 0, 0, 0])
                if qKeyEvent.key() == Qt.Key_S:
                    self.runTrial.trial.agent.take_training_step([-1, 0, 0, 0])
                if qKeyEvent.key() == Qt.Key_A:
                    self.runTrial.trial.agent.take_training_step([0, -1, 0, 0])
                if qKeyEvent.key() == Qt.Key_D:
                    self.runTrial.trial.agent.take_training_step([0, 1, 0, 0])
                if qKeyEvent.key() == Qt.Key_8:
                    self.runTrial.trial.agent.take_training_step([0, 0, 1, 0])
                if qKeyEvent.key() == Qt.Key_2:
                    self.runTrial.trial.agent.take_training_step([0, 0, -1, 0])
                if qKeyEvent.key() == Qt.Key_4:
                    self.runTrial.trial.agent.take_training_step([0, 0, 0, 1])
                if qKeyEvent.key() == Qt.Key_6:
                    self.runTrial.trial.agent.take_training_step([0, 0, 0, -1])
            else:
                if qKeyEvent.key() == Qt.Key_W:
                    self.runTrial.trial.agent.take_training_step([1, 0, 0])
                if qKeyEvent.key() == Qt.Key_S:
                    self.runTrial.trial.agent.take_training_step([-1, 0, 0])
                if qKeyEvent.key() == Qt.Key_A:
                    self.runTrial.trial.agent.take_training_step([0, -1, 0])
                if qKeyEvent.key() == Qt.Key_D:
                    self.runTrial.trial.agent.take_training_step([0, 1, 0])
                if qKeyEvent.key() == Qt.Key_8:
                    self.runTrial.trial.agent.take_training_step([0, 0, 1])
                if qKeyEvent.key() == Qt.Key_2:
                    self.runTrial.trial.agent.take_training_step([0, 0, -1])

        if self.runTrial.trial.agent.end:
            if qKeyEvent.key() == Qt.Key_Escape:
                self.close()

    def controlBehaviour(self):
        if self.sender().objectName() == 'btnStart':
            self.lblDemoInstructionsTitle.setText('Demonstration Disabled')
            self.lblDemoInstructionsTitle.setStyleSheet('color: red; font-size: 13pt')
            self.frameFeedback.setEnabled(True)
            self.runTrial.trial.agent.startingPoint = False
            self.runTrial.trial.message['command'] = 'start'
        elif self.sender().objectName() == 'btnPause':
            self.lblDemoInstructionsTitle.setText('Demonstration Enabled')
            self.lblDemoInstructionsTitle.setStyleSheet('color: green; font-size: 13pt')
            self.frameFeedback.setEnabled(False)
            self.runTrial.trial.agent.startingPoint = False
            self.runTrial.trial.message['command'] = 'pause'
        elif self.sender().objectName() == 'btnStop':
            self.runTrial.trial.message['command'] = 'stop'
            self.close()
        elif self.sender().objectName() == 'btnReset':
            if not self.runTrial.trial.trainingPhase:
                self.lblDemoInstructionsTitle.setText('Select Starting Point')
                self.lblDemoInstructionsTitle.setStyleSheet('color: purple; font-size: 13pt')
                self.frameFeedback.setEnabled(False)
                self.runTrial.trial.agent.startingPoint = True
                self.runTrial.trial.message['command'] = 'reset'
            else:
                self.runTrial.trial.agent.env.reset(seed=self.runTrial.trial.agent.seedNumber)
        
    def setLblText(self, render):
        self.lblGameInstructions.setText(render['display']['Game Instructions'])

        # CHECKING GAME
        if self.runTrial.trial.gameName == "PandaPickAndPlaceDense-v3":
            self.lblDemoInstructionsGrip.setText('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Document</title></head><body ><table style="text-align: center;"><tr><td></td><td style="text-align: center;"><img src=":Icons/icons/key_8.png"/></td><tr><td></td><td><p style="text-align: center;">Up</p></td></tr></tr>   <tr><td style="text-align: center;"><img src=":/Icons/icons/key_4.png"/></td><td style="text-align: center;"><img src=":/Icons/icons/key_2.png"/></td><td style="text-align: center;"><img src=":/Icons/icons/key_6.png"/></td></tr><tr><td><p>Open Grip</p></td><td><p style="margin-left: 15px; margin-right: 15px;">Down</p></td><td><p>Close Grip</p></td></tr></table> </body></html>')
        
        # CHECKING TASK
        if self.runTrial.trial.taskName == "CustomReachEnv1":
            self.lblCustomTask.setText("start -> green")
            self.lblCustomTask.setStyleSheet('color: green; font-size: 13pt')
        elif self.runTrial.trial.taskName == "CustomReachEnv2":
            self.lblCustomTask.setText("start -> green (x - red)")
            self.lblCustomTask.setStyleSheet('color: Red; font-size: 13pt')
        elif self.runTrial.trial.taskName == "CustomReachEnv3":
            self.lblCustomTask.setText("start -> blue -> green")
            self.lblCustomTask.setStyleSheet('color: blue; font-size: 13pt')

        # CHECKING TRAINING PHASE AND STARTING POINT
        if self.runTrial.trial.trainingPhase:
            self.lblDemoInstructionsTitle.setText('Training Phase')
            self.lblDemoInstructionsTitle.setStyleSheet('color: blue; font-size: 13pt')
        elif self.runTrial.trial.agent.startingPoint:
            self.lblDemoInstructionsTitle.setText('Select Starting Point')
            self.lblDemoInstructionsTitle.setStyleSheet('color: purple; font-size: 13pt')
            self.frameFeedback.setEnabled(False)

        # SET FEEDBACK AND DEMOS LEFT
        self.pbDemo.setValue(int(render['display']['Demos Left']))
        self.pbFeedback.setValue(int(render['display']['Feedback Left']))

        # END GAME
        if self.runTrial.trial.agent.end:
            self.lblDemoInstructionsTitle.setText("PRESS ESC TO END")
            self.lblDemoInstructionsTitle.setStyleSheet('color: black; font-size: 13pt')
            self.btnReset.setEnabled(False)
            self.btnPause.setEnabled(False)
            self.btnStart.setEnabled(False)
            self.runTrial.trial.agent.startingPoint = False
            if self.runTrial.trial.agent.trial_timer >= 180:
                self.lblCustomTask.setText("Time Limit Exceeded")
                self.lblCustomTask.setStyleSheet('color: black; font-size: 13pt')


    def getFeedback(self):
        if self.sender().objectName() == 'btnFeedbackGood':
            self.feedback = 'good'
        elif self.sender().objectName() == 'btnFeedbackBad':
            self.feedback =  'bad'
        self.runTrial.trial.message['feedback'] = self.feedback

    def getTrainingPhase(self):
        if self.runTrial.trial.trainingPhase:
            self.btnStart.setEnabled(False)
            self.btnPause.setEnabled(False)
            self.frameSteps_Feedback.setEnabled(False)
        else:
            self.lblDemoInstructionsTitle.setText('Select Starting Point')
            self.lblDemoInstructionsTitle.setStyleSheet('color: purple; font-size: 13pt')
            self.frameFeedback.setEnabled(False)
            self.btnStart.setEnabled(True)
            self.btnPause.setEnabled(True)
            self.frameSteps_Feedback.setEnabled(True)
            self.runTrial.trial.agent.env.reset(seed=self.runTrial.trial.agent.seedNumber)
            self.btnNext.setEnabled(False)

    def trainingPhase(self):
        if self.runTrial.trial.phase != 4:
            self.runTrial.trial.phase += 1
            self.runTrial.trial.send_render()
        else:
            self.runTrial.trial.trainingPhase = False
            self.runTrial.trial.phase = 5
            self.runTrial.trial.send_render()
            self.getTrainingPhase()


if __name__ == '__main__':
    pass