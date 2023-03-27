from PyQt5.QtWidgets import *
import sys
import cv2
import numpy as np

class GrabCut(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image cutting")
        self.setGeometry(200, 200, 700, 200)

        fileButton = QPushButton("File", self)
        paintButton = QPushButton("Painting", self)
        cutButton = QPushButton("Cut", self)
        incButton = QPushButton("+", self)
        decButton = QPushButton("-", self)
        saveButton = QPushButton("Frame Save", self)
        quitButton = QPushButton("Exit", self)

        fileButton.setGeometry(10, 10, 100, 30)
        paintButton.setGeometry(110, 10, 100, 30)
        cutButton.setGeometry(210, 10, 100, 30)
        incButton.setGeometry(310, 10, 50, 30)
        decButton.setGeometry(360, 10, 50, 30)
        saveButton.setGeometry(410, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)        

        fileButton.clicked.connect(self.fileOpenFunction)
        paintButton.clicked.connect(self.paintFunction)
        cutButton.clicked.connect(self.cutFunction)
        incButton.clicked.connect(self.incFunction)
        decButton.clicked.connect(self.descFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.BrushSize = 5 # 브러쉬 크기
        self.LColor, self.RColor = (255, 0, 0), (0, 0, 255)


    def fileOpenFunction(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "./")
        self.img = cv2.imread(fname[0])
        if self.img is None:
            sys.exit("Cannot find file")

        self.img_show = np.copy(self.img)
        cv2.imshow("Painting", self.img_show)

        self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        self.mask[:, :] = cv2.GC_PR_BGD

    def paintFunction(self):
        cv2.setMouseCallback("Painting", self.painting)

    def painting(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽버튼 클릭 이벤트
            cv2.circle(self.img_show, (x, y), self.BrushSize, self.LColor, -1) ## 왼쪽버튼 클릭하면 파란색
            cv2.circle(self.mask, (x, y), self.BrushSize, cv2.GC_FGD, -1)

        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(self.img_show, (x, y), self.BrushSize, self.RColor, -1)
            cv2.circle(self.mask, (x, y), self.BrushSize, cv2.GC_FGD, -1)

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON: ## 왼쪽버튼 눌린채로 마우스 이동 이벤트      
            cv2.circle(self.img_show, (x, y), self.BrushSize, self.LColor, -1) ## 왼쪽버튼 클릭하고 이동하면 파란색
            cv2.circle(self.mask, (x, y), self.BrushSize, cv2.GC_FGD, -1)

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:            
            cv2.circle(self.img_show, (x, y), self.BrushSize, self.RColor, -1)
            cv2.circle(self.mask, (x, y), self.BrushSize, cv2.GC_FGD, -1)
         
        cv2.imshow("Painting", self.img_show)


    def cutFunction(self):
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)

        cv2.grabCut(self.img, self.mask, None, background, foreground, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask ==2) | (self.mask == 0), 0, 1).astype('uint8')
        self.grabImg = self.img * mask2[:, :, np.newaxis]

        cv2.imshow("Scissoring", self.grabImg)

    def incFunction(self):
        self.BrushSize = min(20, self.BrushSize + 1)

    def descFunction(self):
        self.BrushSize = max(1, self.BrushSize - 1)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, "File save", "./")
        cv2.imwrite(fname[0], self.grabImg)

    def quitFunction(self):
        cv2.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = GrabCut()
win.show()
app.exec_()
    




