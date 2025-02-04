from PyQt5.QtWidgets import QLabel


class VideoLabel(QLabel):
    def __init__(self, parent=None, width=640, height=360):
        super().__init__(parent)
        self.setFixedSize(width, height)
