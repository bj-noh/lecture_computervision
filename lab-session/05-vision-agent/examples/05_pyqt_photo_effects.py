from __future__ import annotations

import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from qt_cv_utils import cv_to_qpixmap, data_dir, output_dir


class PhotoEffectsAgent(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vision Agent - Photo Effects")
        self.setGeometry(140, 140, 980, 760)

        self.original = cv2.imread(str(data_dir() / "cat.jpg"))
        self.current = self.original.copy() if self.original is not None else None

        self.image_label = QLabel("Load an image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.status_label = QLabel("Choose an effect and save the result if needed.")

        open_button = QPushButton("Open Image")
        gray_button = QPushButton("Gray")
        sketch_button = QPushButton("Sketch")
        edge_button = QPushButton("Edges")
        cartoon_button = QPushButton("Cartoon")
        reset_button = QPushButton("Reset")
        save_button = QPushButton("Save")

        open_button.clicked.connect(self.open_image)
        gray_button.clicked.connect(lambda: self.apply_effect("gray"))
        sketch_button.clicked.connect(lambda: self.apply_effect("sketch"))
        edge_button.clicked.connect(lambda: self.apply_effect("edges"))
        cartoon_button.clicked.connect(lambda: self.apply_effect("cartoon"))
        reset_button.clicked.connect(self.reset_image)
        save_button.clicked.connect(self.save_image)

        row = QHBoxLayout()
        for button in [open_button, gray_button, sketch_button, edge_button, cartoon_button, reset_button, save_button]:
            row.addWidget(button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addLayout(row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.refresh_view()

    def open_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "./")
        if filename:
            loaded = cv2.imread(filename)
            if loaded is not None:
                self.original = loaded
                self.current = loaded.copy()
                self.status_label.setText(f"Loaded: {filename}")
                self.refresh_view()

    def apply_effect(self, mode: str) -> None:
        if self.original is None:
            self.status_label.setText("Open an image first")
            return

        image = self.original.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if mode == "gray":
            self.current = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif mode == "sketch":
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            inverted = cv2.divide(gray, 255 - blur, scale=256)
            self.current = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
        elif mode == "edges":
            edges = cv2.Canny(gray, 80, 160)
            self.current = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif mode == "cartoon":
            color = cv2.bilateralFilter(image, 9, 75, 75)
            edges = cv2.adaptiveThreshold(
                cv2.medianBlur(gray, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                5,
            )
            self.current = cv2.bitwise_and(color, color, mask=edges)

        self.status_label.setText(f"Applied effect: {mode}")
        self.refresh_view()

    def reset_image(self) -> None:
        if self.original is not None:
            self.current = self.original.copy()
            self.status_label.setText("Image reset")
            self.refresh_view()

    def save_image(self) -> None:
        if self.current is None:
            self.status_label.setText("No image to save")
            return
        default_path = output_dir() / "photo_effect_result.png"
        filename, _ = QFileDialog.getSaveFileName(self, "Save Image", str(default_path))
        if filename:
            cv2.imwrite(filename, self.current)
            self.status_label.setText(f"Saved: {filename}")

    def refresh_view(self) -> None:
        if self.current is None:
            return
        self.image_label.setPixmap(
            cv_to_qpixmap(self.current).scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoEffectsAgent()
    window.show()
    app.exec_()
