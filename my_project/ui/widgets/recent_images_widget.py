from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout


class RecentImagesWidget(QWidget):
    def __init__(self, num_classes, parent=None):
        super().__init__(parent)
        self.num_classes = num_classes
        # 내부 상태: 클래스 1부터 num_classes-1까지의 최근 판별 이미지 목록
        self.recent_images = {cls: [] for cls in range(1, self.num_classes)}
        self.recent_image_labels = {}
        layout = QVBoxLayout()
        # 클래스 0는 PASS 용으로 사용한다고 가정한 뒤, 1부터 시작
        for cls in range(1, self.num_classes):
            row_layout = QHBoxLayout()
            title = QLabel(f"Class {cls}:")
            row_layout.addWidget(title)
            self.recent_image_labels[cls] = []
            for i in range(5):
                thumb = QLabel()
                thumb.setFixedSize(80, 60)
                thumb.setStyleSheet("border: 1px solid black;")
                row_layout.addWidget(thumb)
                self.recent_image_labels[cls].append(thumb)
            layout.addLayout(row_layout)
        self.setLayout(layout)

    def update_for_class(self, cls, pixmap):
        if cls not in self.recent_images:
            return
        self.recent_images[cls].insert(0, pixmap)
        if len(self.recent_images[cls]) > 5:
            self.recent_images[cls].pop()
        for i, label in enumerate(self.recent_image_labels[cls]):
            if i < len(self.recent_images[cls]):
                label.setPixmap(self.recent_images[cls][i])
            else:
                label.clear()
