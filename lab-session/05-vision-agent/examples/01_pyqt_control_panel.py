from __future__ import annotations

import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget


class VisionControlPanel(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Vision Agent Control Panel")
        self.setGeometry(200, 200, 480, 180)

        self.label = QLabel("PyQt control panel for simple vision-agent feedback.")

        short_button = QPushButton("Short Alert")
        long_button = QPushButton("Long Alert")
        clear_button = QPushButton("Clear")
        quit_button = QPushButton("Exit")

        short_button.clicked.connect(self.short_alert)
        long_button.clicked.connect(self.long_alert)
        clear_button.clicked.connect(self.clear_status)
        quit_button.clicked.connect(self.close)

        button_row = QHBoxLayout()
        button_row.addWidget(short_button)
        button_row.addWidget(long_button)
        button_row.addWidget(clear_button)
        button_row.addWidget(quit_button)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(button_row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def short_alert(self) -> None:
        self.label.setText("Short alert: a brief warning was triggered.")
        QApplication.beep()

    def long_alert(self) -> None:
        self.label.setText("Long alert: repeated warning beeps are active.")
        for step in range(4):
            QTimer.singleShot(250 * step, QApplication.beep)

    def clear_status(self) -> None:
        self.label.setText("Status cleared. Waiting for the next event.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionControlPanel()
    window.show()
    app.exec_()
