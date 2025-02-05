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
        num_classes: 전체 클래스 개수 (여기서는 0: NG1, 1: NG2, 2: GOOD)
                   하지만 GOOD은 표시하지 않으므로, RecentImagesWidget에서는 NG1과 NG2만 사용.
        thumbnail_size: 각 썸네일의 크기를 지정 (QSize 객체)
        max_images_per_class: 각 클래스마다 최대 저장할 최근 이미지 수
        """
        super().__init__(parent)
        # 좋은(=GOOD, 인덱스 2)는 표시하지 않으므로, 사용 클래스는 0과 1만 사용함
        self.num_display_classes = 2
        self.thumbnail_size = thumbnail_size
        self.max_images_per_class = max_images_per_class

        # 각 클래스별 최근 이미지들을 저장 (여기서는 0: NG1, 1: NG2)
        self.recent_images = {cls: [] for cls in range(self.num_display_classes)}
        self.recent_image_labels = {}  # 각 클래스에 해당하는 QLabel 리스트

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # 클래스 0과 1만 사용: 0 -> NG1, 1 -> NG2
        for cls in range(self.num_display_classes):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(2)
            row_layout.setContentsMargins(0, 0, 0, 0)

            if cls == 0:
                title_text = "NG1:"
            elif cls == 1:
                title_text = "NG2:"
            else:
                title_text = f"Class {cls}:"
            title = QLabel(title_text)
            title.setContentsMargins(0, 0, 0, 0)
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
        GOOD(클래스 2)은 표시하지 않습니다.
        """
        # GOOD(클래스 2)는 Recent 이미지는 표시하지 않음
        if cls == 2:
            return

        if pixmap.isNull():
            return

        # pixmap 크기가 thumbnail_size와 다르면 스케일링
        if pixmap.size() != self.thumbnail_size:
            pixmap = pixmap.scaled(
                self.thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        if cls not in self.recent_images:
            return

        # 최신 이미지를 맨 앞에 추가 후, 최대 저장 개수 초과 시 마지막 이미지 삭제
        self.recent_images[cls].insert(0, pixmap)
        if len(self.recent_images[cls]) > self.max_images_per_class:
            self.recent_images[cls].pop()

        # 각 QLabel을 최신 이미지 순서로 업데이트
        for i, label in enumerate(self.recent_image_labels[cls]):
            if i < len(self.recent_images[cls]):
                label.setPixmap(self.recent_images[cls][i])
            else:
                label.clear()
