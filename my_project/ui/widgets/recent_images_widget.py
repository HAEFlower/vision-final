from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtCore import QSize, Qt


class RecentImagesWidget(QWidget):
    def __init__(
        self,
        num_classes,
        thumbnail_size=QSize(50, 140),
        max_images_per_class=5,
        parent=None,
    ):
        """
        num_classes: 전체 클래스 개수. (클래스 0은 PASS 용도로 사용한다고 가정하여 1부터 시작)
        thumbnail_size: 각 썸네일의 크기를 지정 (QSize 객체)
        max_images_per_class: 각 클래스마다 최대 저장할 최근 이미지 수
        """
        super().__init__(parent)
        self.num_classes = num_classes
        self.thumbnail_size = thumbnail_size
        self.max_images_per_class = max_images_per_class

        # 각 클래스별 최근 이미지들을 저장 (클래스 0은 PASS 용으로 사용됨; 1부터 시작)
        self.recent_images = {cls: [] for cls in range(1, self.num_classes)}
        self.recent_image_labels = {}  # 각 클래스에 해당하는 QLabel 리스트

        main_layout = QVBoxLayout()
        # 전체 레이아웃의 여백과 간격을 설정합니다.
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # 클래스 0은 PASS용이고, 1부터 시작합니다.
        for cls in range(1, self.num_classes):
            row_layout = QHBoxLayout()
            # 텍스트와 이미지 사이의 간격을 줄입니다.
            row_layout.setSpacing(2)
            row_layout.setContentsMargins(0, 0, 0, 0)

            title = QLabel(f"Class {cls}:")
            title.setContentsMargins(0, 0, 0, 0)
            # title이 불필요하게 확장되지 않도록 설정
            title.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            row_layout.addWidget(title)

            self.recent_image_labels[cls] = []
            for _ in range(self.max_images_per_class):
                thumb = QLabel()
                thumb.setFixedSize(self.thumbnail_size)
                thumb.setStyleSheet("border: 2px solid black;")
                row_layout.addWidget(thumb)
                self.recent_image_labels[cls].append(thumb)
            main_layout.addLayout(row_layout)

        self.setLayout(main_layout)

    def update_for_class(self, cls, pixmap):
        """
        지정된 클래스로 새 pixmap을 추가하고, 최신 이미지 순서로 표시합니다.
        전달받은 pixmap의 크기가 thumbnail_size와 다르면 자동으로 스케일링합니다.
        """
        if pixmap.isNull():
            return

        # pixmap 크기가 다르면 thumbnail_size에 맞춰 자동 스케일링
        if pixmap.size() != self.thumbnail_size:
            pixmap = pixmap.scaled(
                self.thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        if cls not in self.recent_images:
            return

        # 최신 이미지를 맨 앞에 추가하고, 최대 저장 개수를 초과하면 마지막 이미지를 삭제
        self.recent_images[cls].insert(0, pixmap)
        if len(self.recent_images[cls]) > self.max_images_per_class:
            self.recent_images[cls].pop()

        # 각 QLabel에 대해 최신 이미지 순서로 업데이트합니다.
        for i, label in enumerate(self.recent_image_labels[cls]):
            if i < len(self.recent_images[cls]):
                label.setPixmap(self.recent_images[cls][i])
            else:
                label.clear()
