from PyQt5.QtWidgets import *
import sys
import cv2

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video frame capture")
        self.setGeometry(200, 200, 500, 100)

        videoButton = QPushButton("Video On", self)
        captureButton = QPushButton("Frame Capture", self)
        saveButton = QPushButton("Frame Save", self)
        quitButton = QPushButton("Exit", self)

        videoButton.setGeometry(10, 10, 100, 30)
        captureButton.setGeometry(110, 10, 100, 30)
        saveButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(310, 10, 100, 30)
        

        videoButton.clicked.connect(self.videoFunction)
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def videoFunction(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            self.close()

        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                break

            cv2.imshow("Video display", self.frame)
            cv2.waitKey(1)

    def captureFunction(self):
        self.capturedFrame = self.frame
        cv2.imshow("Captured frame", self.capturedFrame)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, "File save", "./")
        cv2.imwrite(fname[0], self.capturedFrame)

    def quitFunction(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = Video()
win.show()
app.exec_()
    




