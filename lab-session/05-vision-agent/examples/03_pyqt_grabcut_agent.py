from __future__ import annotations

import sys

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from qt_cv_utils import output_dir


class GrabCutAgent(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vision Agent - GrabCut")
        self.setGeometry(160, 160, 760, 180)

        self.image = None
        self.image_view = None
        self.mask = None
        self.result = None
        self.brush_size = 6

        self.status_label = QLabel("Open an image, paint foreground/background, then run GrabCut.")

        open_button = QPushButton("Open Image")
        paint_button = QPushButton("Enable Painting")
        cut_button = QPushButton("Run GrabCut")
        inc_button = QPushButton("Brush +")
        dec_button = QPushButton("Brush -")
        save_button = QPushButton("Save Result")
        exit_button = QPushButton("Exit")

        open_button.clicked.connect(self.open_image)
        paint_button.clicked.connect(self.enable_painting)
        cut_button.clicked.connect(self.run_grabcut)
        inc_button.clicked.connect(self.increase_brush)
        dec_button.clicked.connect(self.decrease_brush)
        save_button.clicked.connect(self.save_result)
        exit_button.clicked.connect(self.close)

        row = QHBoxLayout()
        for button in [open_button, paint_button, cut_button, inc_button, dec_button, save_button, exit_button]:
            row.addWidget(button)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addLayout(row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "./")
        if not filename:
            return

        self.image = cv2.imread(filename)
        if self.image is None:
            self.status_label.setText("Could not open the selected file")
            return

        self.image_view = self.image.copy()
        self.mask = np.full(self.image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        cv2.imshow("GrabCut Painting", self.image_view)
        self.status_label.setText("Image loaded. Left drag=foreground, right drag=background")

    def enable_painting(self) -> None:
        if self.image_view is None:
            self.status_label.setText("Open an image first")
            return
        cv2.setMouseCallback("GrabCut Painting", self.painting)
        self.status_label.setText("Painting enabled")

    def painting(self, event, x, y, flags, param) -> None:
        del param
        if self.image_view is None or self.mask is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            cv2.circle(self.image_view, (x, y), self.brush_size, (255, 0, 0), -1)
            cv2.circle(self.mask, (x, y), self.brush_size, cv2.GC_FGD, -1)
        elif event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON):
            cv2.circle(self.image_view, (x, y), self.brush_size, (0, 0, 255), -1)
            cv2.circle(self.mask, (x, y), self.brush_size, cv2.GC_BGD, -1)

        cv2.imshow("GrabCut Painting", self.image_view)

    def run_grabcut(self) -> None:
        if self.image is None or self.mask is None:
            self.status_label.setText("Open and paint an image first")
            return

        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        cv2.grabCut(self.image, self.mask, None, background, foreground, 5, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where(
            (self.mask == cv2.GC_BGD) | (self.mask == cv2.GC_PR_BGD),
            0,
            1,
        ).astype("uint8")
        self.result = self.image * mask2[:, :, np.newaxis]
        cv2.imshow("GrabCut Result", self.result)
        self.status_label.setText("GrabCut result updated")

    def increase_brush(self) -> None:
        self.brush_size = min(30, self.brush_size + 1)
        self.status_label.setText(f"Brush size: {self.brush_size}")

    def decrease_brush(self) -> None:
        self.brush_size = max(1, self.brush_size - 1)
        self.status_label.setText(f"Brush size: {self.brush_size}")

    def save_result(self) -> None:
        if self.result is None:
            self.status_label.setText("Run GrabCut before saving")
            return
        default_path = output_dir() / "grabcut_result.png"
        filename, _ = QFileDialog.getSaveFileName(self, "Save Result", str(default_path))
        if filename:
            cv2.imwrite(filename, self.result)
            self.status_label.setText(f"Saved: {filename}")

    def closeEvent(self, event) -> None:
        cv2.destroyAllWindows()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GrabCutAgent()
    window.show()
    app.exec_()
