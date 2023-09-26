from GUI.gui import UI
from PyQt5.QtWidgets import QApplication
import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QComboBox, QSpinBox
from PyQt5 import uic


class Start(QMainWindow):
    def __init__(self):
        super(Start, self).__init__()

        # Load the UI file
        uic.loadUi('./GUI/start.ui', self)

        # Define our widgets
        self.btnStart = self.findChild(QPushButton, "btnStart")
        self.drpTask = self.findChild(QComboBox, "drpTask")
        self.txtID = self.findChild(QSpinBox, "txtID")
        self.UIWindow = None

        self.btnStart.clicked.connect(self.startWindow)

        self.show()

    def startWindow(self):
        participantID = self.txtID.text()
        task = self.drpTask.currentText()
        if "Task 0" in task:
            taskName = 'CustomReachEnv0'
        elif "Task 1" in task:
            taskName = 'CustomReachEnv1'
        elif "Task 2" in task:
            taskName = 'CustomReachEnv2'
        elif "Task 3" in task:
            taskName = 'CustomReachEnv3'
        
        if self.UIWindow is not None:
            self.UIWindow.close()
        self.UIWindow = UI(participantID, taskName)
        self.UIWindow.setWindowTitle("Feedback Learning")
        

if __name__ == '__main__':
    start = QApplication(sys.argv)
    UIWindowStart = Start()
    UIWindowStart.setWindowTitle("Feedback Learning")
    start.exec()