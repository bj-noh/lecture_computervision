from PyQt5.QtWidgets import *
import sys
import winsound

class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beep sound")
        self.setGeometry(200, 200, 500, 100)

        shortBeepButton = QPushButton("Short beep", self)
        longBeepButton = QPushButton("Long beep", self)
        quitButton = QPushButton("Exit", self)
        self.label = QLabel("Hello PyQt World", self)

        shortBeepButton.setGeometry(10, 10, 100, 30)
        longBeepButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 70)

        shortBeepButton.clicked.connect(self.shortBeepFunction)
        longBeepButton.clicked.connect(self.longBeepFunction)
        quitButton.clicked.connect(self.quitFunction)

    def shortBeepFunction(self):
        self.label.setText("0.5s beep with Freq 1000")
        winsound.Beep(1000, 500)

    def longBeepFunction(self):
        self.label.setText("3s beep with Freq 1000")
        winsound.Beep(1000, 3000)

    def quitFunction(self):
        self.close()


app = QApplication(sys.argv)
win = BeepSound()
win.show()
app.exec_()
    




