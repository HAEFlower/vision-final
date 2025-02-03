from PyQt5.QtWidgets import QLabel

width = 640
height = 360


class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
