from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QFont


class ResultLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Result: -")
        result_font = QFont("Arial", 48, QFont.Bold)
        self.setFont(result_font)
