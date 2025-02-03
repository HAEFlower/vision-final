import sys
import cv2
import os
import torch
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QApplication
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtGui import QImage, QPixmap
from ui.widgets.video_label import VideoLabel
from ui.widgets.time_label import TimeLabel
from ui.widgets.result_label import ResultLabel
from ui.widgets.stats_label import StatsLabel
from ui.widgets.recent_images_widget import RecentImagesWidget
from ui.widgets.roi_label import ROILabel  # ROI를 그릴 수 있는 QLabel subclass
from model.yolo_model import load_model, run_inference

# 사전 로드된 YOLO 모델
yolo_model = load_model()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 예시: 클래스 수를 3으로 설정 (필요에 따라 수정)
        self.num_classes = 3
        # ROI 영역 내에 객체가 차지해야 하는 최소 비율 (0.6 = 60%)
        self.roi_threshold = 0.6

        self.initialize_window()
        self.setup_ui_components()
        self.initialize_variables()
        self.setup_webcam()
        self.setup_timers()
        self.update_time()

    def initialize_window(self):
        self.setWindowTitle("Webcam & Inference Viewer")
        self.setGeometry(100, 100, 1000, 600)

    def setup_ui_components(self):
        # 왼쪽 영역: 시간, 영상, 결과 표시
        self.time_label = TimeLabel()

        # video_label: 영상 프레임을 표시하는 QLabel
        self.video_label = VideoLabel()
        # ROI 기능 추가: video_label 위에 ROILabel을 자식으로 생성 (투명 overlay)
        self.roi_label = ROILabel(self.video_label)
        self.roi_label.setStyleSheet("background: transparent;")

        self.result_label = ResultLabel()

        self.left_layout = QVBoxLayout()
        self.left_layout.setSpacing(5)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.addWidget(self.time_label)
        self.left_layout.addWidget(self.video_label)
        self.left_layout.addWidget(self.result_label)
        self.left_container = QWidget()
        self.left_container.setLayout(self.left_layout)

        # 오른쪽 영역: 통계 정보와 최근 판별 이미지
        self.stats_label = StatsLabel(self.num_classes)
        self.recent_images_widget = RecentImagesWidget(self.num_classes)
        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.stats_label)
        self.right_layout.addWidget(self.recent_images_widget)
        self.right_container = QWidget()
        self.right_container.setLayout(self.right_layout)

        # 전체 레이아웃
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.left_container)
        main_layout.addWidget(self.right_container)
        self.setLayout(main_layout)

    def initialize_variables(self):
        self.total_count = 0
        self.class_counts = {cls: 0 for cls in range(self.num_classes)}
        self.frame_counter = 0

    def setup_webcam(self):
        self.cap = cv2.VideoCapture(0)

    def setup_timers(self):
        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.update_frame)
        self.timer_video.start(30)  # 약 33 FPS

        self.timer_time = QTimer()
        self.timer_time.timeout.connect(self.update_time)
        self.timer_time.start(1000)  # 1초마다

    def update_time(self):
        now = QDateTime.currentDateTime().toString("yyyy/MM/dd hh:mm:ss")
        self.time_label.setText(now)

    def update_video_display(self, frame):
        # display_frame: 640x360 크기로 리사이즈 후 RGB 변환
        display_frame = cv2.resize(frame, (640, 360))
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame.shape
        bytes_line = ch * w

        qt_img = QImage(display_frame.data, w, h, bytes_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        # video_label 크기에 맞춰 roi_label의 영역 갱신
        self.roi_label.setGeometry(0, 0, w, h)
        return display_frame

    def update_stats_display(self):
        self.stats_label.update_text(self.total_count, self.class_counts)

    def update_stats_and_recent(self, inferred_class, display_frame):
        self.total_count += 1
        self.class_counts[inferred_class] += 1
        self.update_stats_display()

        if inferred_class != 0:
            thumbnail = cv2.resize(display_frame, (80, 60))
            thumb_qimg = QImage(thumbnail.data, 80, 60, 80 * 3, QImage.Format_RGB888)
            thumb_pixmap = QPixmap.fromImage(thumb_qimg)
            self.recent_images_widget.update_for_class(inferred_class, thumb_pixmap)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 좌우 미러링 (사용자에게 친숙한 화면 제공)
        frame = cv2.flip(frame, 1)

        # 전체 프레임을 640x360으로 리사이즈
        display_frame = cv2.resize(frame, (640, 360))
        disp_h, disp_w, _ = display_frame.shape

        # ROI 영역 (ROILabel에서 지정된 영역; display_frame 크기에 맞춤)
        roi = self.roi_label.roi_rect
        if roi is not None:
            roi_x1 = max(0, roi.x())
            roi_y1 = max(0, roi.y())
            roi_x2 = min(disp_w, roi.x() + roi.width())
            roi_y2 = min(disp_h, roi.y() + roi.height())

        # 전체 화면(display_frame)에 대해 inference 실행
        try:
            results = run_inference(yolo_model, display_frame)
        except Exception as e:
            print("YOLO detection error:", e)
            results = None

        annotated_frame = display_frame.copy()
        valid_detection = False

        if results is not None and results[0].boxes is not None:
            # 결과에서 박스, 신뢰도, 클래스 아이디를 numpy 배열로 변환
            try:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy()
            except Exception:
                boxes = results[0].boxes.xyxy.numpy()
                confs = results[0].boxes.conf.numpy()
                cls_ids = results[0].boxes.cls.numpy()

            # 각 검출 박스에 대해 ROI와의 겹침(Overlap) 비율 계산 및 표시
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                detection_area = max(1, (x2 - x1) * (y2 - y1))
                if roi is not None:
                    inter_x1 = max(x1, roi_x1)
                    inter_y1 = max(y1, roi_y1)
                    inter_x2 = min(x2, roi_x2)
                    inter_y2 = min(y2, roi_y2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    intersection_area = inter_w * inter_h
                    ratio = intersection_area / detection_area
                else:
                    ratio = 1.0

                # roi_threshold (예: 0.6) 이상인 경우에만 valid로 판단
                if ratio >= self.roi_threshold:
                    valid_detection = True

                    # 박스 그리기 (색: 초록, 두께: 2)
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )

                    # 클래스 이름 및 신뢰도 표시를 위한 폰트 설정
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1

                    # 클래스 이름 (왼쪽 상단)
                    class_id = int(cls_ids[i])
                    conf = confs[i]
                    if hasattr(yolo_model.model, "names"):
                        class_name = yolo_model.model.names.get(class_id, "Unknown")
                    else:
                        class_name = str(class_id)
                    (text_w, text_h), baseline = cv2.getTextSize(
                        class_name, font, font_scale, thickness
                    )
                    # 배경 사각형 (클래스 이름 표시 영역)
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1) - text_h - baseline),
                        (int(x1) + text_w, int(y1)),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        class_name,
                        (int(x1), int(y1) - baseline),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA,
                    )

                    # 신뢰도 (오른쪽 상단)
                    conf_str = f"{conf:.2f}"
                    (conf_w, conf_h), conf_baseline = cv2.getTextSize(
                        conf_str, font, font_scale, thickness
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (int(x2) - conf_w, int(y1) - conf_h - conf_baseline),
                        (int(x2), int(y1)),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        conf_str,
                        (int(x2) - conf_w, int(y1) - conf_baseline),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                        cv2.LINE_AA,
                    )

        # 화면에 annotated_frame을 업데이트
        final_display = self.update_video_display(annotated_frame)

        # 결과 라벨 업데이트: 유효한 검출이 있으면 "Detected", 없으면 "No Detection"
        if valid_detection:
            self.result_label.setText("Detected")
            self.result_label.setStyleSheet("color: green;")
            inferred_class = 1
        else:
            self.result_label.setText("No Detection")
            self.result_label.setStyleSheet("color: green;")
            inferred_class = 0

        self.frame_counter += 1
        if self.frame_counter % 30 == 0:
            self.update_stats_and_recent(inferred_class, final_display)

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
