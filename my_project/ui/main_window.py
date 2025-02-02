import sys
import random
import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QApplication
from PyQt5.QtCore import QTimer, QDateTime, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam & Inference Viewer")
        self.setGeometry(100, 100, 1000, 600)

        # --- 왼쪽 영역 ---
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.time_label.setFont(QFont("Arial", 10))
        self.time_label.setFixedHeight(20)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 360)

        self.result_label = QLabel("Result: -")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_font = QFont("Arial", 48, QFont.Bold)
        self.result_label.setFont(result_font)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.time_label)
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.result_label)
        left_container = QWidget()
        left_container.setLayout(left_layout)

        # --- 오른쪽 영역 ---
        # 오른쪽 상단: 통계 정보 (총 건수와 클래스별 판별 수 + %)
        self.stats_label = QLabel()
        self.stats_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # RichText를 사용해서 HTML 태그 적용
        self.stats_label.setTextFormat(Qt.RichText)
        # 초기 통계 텍스트 설정 (나중에 update_stats_display()에서 갱신)
        self.stats_label.setText(
            "<p style='font-size:18pt; font-weight:bold; margin:0;'>Total: 0</p>"
            "<p style='font-size:14pt; margin:0;'>Class 0: 0 (0.0%)</p>"
            "<p style='font-size:14pt; margin:0;'>Class 1: 0 (0.0%)</p>"
            "<p style='font-size:14pt; margin:0;'>Class 2: 0 (0.0%)</p>"
            "<p style='font-size:14pt; margin:0;'>Class 3: 0 (0.0%)</p>"
            "<p style='font-size:14pt; margin:0;'>Class 4: 0 (0.0%)</p>"
        )

        # 오른쪽 하단: 클래스 1~4 최신 판별 이미지 (각 5개)
        self.recent_images_widget = QWidget()
        self.recent_images_layout = QVBoxLayout()
        self.recent_image_labels = {}
        for cls in range(1, 5):
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
            self.recent_images_layout.addLayout(row_layout)
        self.recent_images_widget.setLayout(self.recent_images_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.stats_label)
        right_layout.addWidget(self.recent_images_widget)
        right_container = QWidget()
        right_container.setLayout(right_layout)

        # --- 전체 레이아웃 (왼쪽과 오른쪽) ---
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_container)
        main_layout.addWidget(right_container)
        self.setLayout(main_layout)

        # --- 초기화 (웹캠, 통계 관련 변수) ---
        self.cap = cv2.VideoCapture(0)
        self.total_count = 0
        # 클래스별 (0~4) 판별 횟수
        self.class_counts = {cls: 0 for cls in range(5)}
        self.recent_images = {cls: [] for cls in range(1, 5)}
        self.frame_counter = 0

        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.update_frame)
        self.timer_video.start(30)  # 약 33 FPS

        self.timer_time = QTimer()
        self.timer_time.timeout.connect(self.update_time)
        self.timer_time.start(1000)
        self.update_time()

    def update_time(self):
        now = QDateTime.currentDateTime().toString("yyyy/MM/dd hh:mm:ss")
        self.time_label.setText(now)

    def update_video_display(self, frame):
        display_frame = cv2.resize(frame, (640, 360))
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame.shape
        bytes_line = ch * w
        qt_img = QImage(display_frame.data, w, h, bytes_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        return display_frame

    def perform_inference(self, frame):
        # 예시: 랜덤하게 0~4 범위의 값을 생성 (실제 모델 추론 코드로 대체)
        return random.choices(range(5), weights=[0.7, 0.1, 0.1, 0.05, 0.05])[0]

    def update_stats_display(self):
        # HTML을 사용해 총 건수(total)를 크게, 아래에 각 클래스 판별 수와 백분율을 한 줄씩 표시
        stats_html = f"<p style='font-size:18pt; font-weight:bold; margin:0;'>Total: {self.total_count}</p>"
        for cls in range(5):
            count = self.class_counts[cls]
            percent = (count / self.total_count * 100) if self.total_count > 0 else 0.0
            stats_html += f"<p style='font-size:14pt; margin:0;'>Class {cls}: {count} ({percent:.1f}%)</p>"
        self.stats_label.setText(stats_html)

    def update_stats_and_recent(self, inferred_class, display_frame):
        # 업데이트는 1초에 한 번 (약 30프레임마다)
        self.total_count += 1
        self.class_counts[inferred_class] += 1
        self.update_stats_display()

        # FAIL 결과 (inferred_class != 0)인 경우만 최신 이미지 업데이트 (클래스 1~4)
        if inferred_class != 0 and inferred_class in self.recent_images:
            thumbnail = cv2.resize(display_frame, (80, 60))
            thumb_qimg = QImage(thumbnail.data, 80, 60, 3 * 80, QImage.Format_RGB888)
            thumb_pixmap = QPixmap.fromImage(thumb_qimg)
            recents = self.recent_images[inferred_class]
            recents.insert(0, thumb_pixmap)
            if len(recents) > 5:
                recents.pop()
            for i, label in enumerate(self.recent_image_labels[inferred_class]):
                if i < len(recents):
                    label.setPixmap(recents[i])
                else:
                    label.clear()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)  # 미러링

        display_frame = self.update_video_display(frame)
        inferred_class = self.perform_inference(frame)

        # 왼쪽 하단 큰 PASS/FAIL 표시 (0일 때 PASS, 그 외 FAIL)
        if inferred_class == 0:
            self.result_label.setText("PASS")
            self.result_label.setStyleSheet("color: green;")
        else:
            self.result_label.setText("FAIL")
            self.result_label.setStyleSheet("color: red;")

        self.frame_counter += 1
        if self.frame_counter % 30 == 0:
            self.update_stats_and_recent(inferred_class, display_frame)

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
