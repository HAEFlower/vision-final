import sys
import cv2
import numpy as np
import random  # 임의의 랜덤 리턴을 위해 추가
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QApplication
from PyQt5.QtCore import QTimer, QDateTime, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter
from ui.widgets.video_label import VideoLabel
from ui.widgets.time_label import TimeLabel
from ui.widgets.result_label import ResultLabel
from ui.widgets.stats_label import StatsLabel
from ui.widgets.recent_images_widget import RecentImagesWidget
from ui.widgets.roi_label import ROILabel  # ROI를 그릴 수 있는 QLabel subclass
from model.yolo_model import load_model, run_inference
from utils.sort import Sort

# 사전 로드된 YOLO 모델
yolo_model = load_model()


def create_thumbnail(pixmap, target_size):
    """
    원본 pixmap의 전체 비율을 유지하며 target_size (QSize)로 축소한 후,
    target_size 크기의 빈 캔버스에 중앙에 배치하여 반환하는 함수.
    """
    scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    thumb = QPixmap(target_size)
    thumb.fill(Qt.transparent)
    x = (target_size.width() - scaled.width()) // 2
    y = (target_size.height() - scaled.height()) // 2
    painter = QPainter(thumb)
    painter.drawPixmap(x, y, scaled)
    painter.end()
    return thumb


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 클래스 수 및 ROI 기준 (필요에 따라 수정)
        self.num_classes = 3
        self.roi_threshold = 0.6

        self.tracker = Sort()
        self.track_flags = {}
        # self.track_classifications: track_id -> 현재 예측 (0, 1, 2)
        self.track_classifications = {}
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
        self.video_label = VideoLabel()
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

        # 오른쪽 영역: 통계 정보 및 최근 판별 이미지
        self.stats_label = StatsLabel(self.num_classes)
        self.recent_images_widget = RecentImagesWidget(self.num_classes)
        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.stats_label)
        self.right_layout.addWidget(self.recent_images_widget)
        self.right_container = QWidget()
        self.right_container.setLayout(self.right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.left_container)
        main_layout.addWidget(self.right_container)
        self.setLayout(main_layout)

    def initialize_variables(self):
        # 누적 통계 변수: track이 처음 등장할 때마다 누적 (사라져도 삭제하지 않습니다)
        self.cumulative_total = 0
        self.cumulative_class_counts = {i: 0 for i in range(self.num_classes)}
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
        display_frame = cv2.resize(frame, (640, 360))
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame_rgb.shape
        bytes_line = ch * w
        qt_img = QImage(display_frame_rgb.data, w, h, bytes_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        self.roi_label.setGeometry(0, 0, w, h)
        return display_frame

    def update_stats_display(self):
        # 누적 통계로 업데이트 (track이 한번 등장하면 누적)
        self.stats_label.update_text(
            self.cumulative_total, self.cumulative_class_counts
        )

    def perform_classification(self, crop_img):
        """
        구별 모델 추론 함수
        실제 전처리 및 추론 코드를 추가할 수 있으며,
        현재는 임시로 0, 1, 2 중 랜덤 값을 리턴합니다.
        """
        return random.choice([0, 1, 2])

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 좌우 반전 제거: 원본 프레임 그대로 사용
        # frame = cv2.flip(frame, 1)  <-- 이 줄을 제거했습니다.
        display_frame = cv2.resize(frame, (640, 360))
        # 크롭 시 사용할 BGR 원본 이미지
        orig_frame = display_frame.copy()
        disp_h, disp_w, _ = display_frame.shape

        # ROI 영역 (ROILabel 좌표 기준)
        roi = self.roi_label.roi_rect
        if roi is not None:
            roi_x1 = max(0, roi.x())
            roi_y1 = max(0, roi.y())
            roi_x2 = min(disp_w, roi.x() + roi.width())
            roi_y2 = min(disp_h, roi.y() + roi.height())

        try:
            results = run_inference(yolo_model, display_frame)
        except Exception as e:
            print("YOLO detection error:", e)
            results = None

        valid_det_info = []
        annotated_frame = display_frame.copy()
        if results is not None and results[0].boxes is not None:
            try:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy()
            except Exception:
                boxes = results[0].boxes.xyxy.numpy()
                confs = results[0].boxes.conf.numpy()
                cls_ids = results[0].boxes.cls.numpy()

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
                if ratio >= self.roi_threshold:
                    valid_det_info.append([x1, y1, x2, y2, confs[i], int(cls_ids[i])])
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0),
                        1,
                    )

        if len(valid_det_info) > 0:
            dets = np.array(valid_det_info)[:, :5]
            tracked_objects = self.tracker.update(dets)
        else:
            tracked_objects = np.empty((0, 5))

        valid_detection = False
        for trk in tracked_objects:
            x1, y1, x2, y2, track_id = trk
            track_id = int(track_id)

            best_iou = 0
            best_det = None
            for det in valid_det_info:
                iou_val = self.compute_iou([x1, y1, x2, y2], det[:4])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = det
            if best_iou > 0.5 and best_det is not None:
                det_class = best_det[5]
                if det_class != 0:
                    self.track_flags[track_id] = True
                else:
                    self.track_flags.setdefault(track_id, False)
            else:
                self.track_flags.setdefault(track_id, False)

            if self.track_flags.get(track_id, False):
                valid_detection = True

            # 트랙 박스 및 ID 그리기 (초록색)
            x1_int, y1_int, x2_int, y2_int = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(
                annotated_frame, (x1_int, y1_int), (x2_int, y2_int), (0, 255, 0), 2
            )
            label_id = f"ID:{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label_id, font, font_scale, thickness
            )
            cv2.rectangle(
                annotated_frame,
                (x1_int, y1_int - text_h - baseline),
                (x1_int + text_w, y1_int),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated_frame,
                label_id,
                (x1_int, y1_int - baseline),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

            # 객체 영역을 원본 박스 크기로 크롭하여 구별 모델 추론
            crop_x1 = max(0, x1_int)
            crop_y1 = max(0, y1_int)
            crop_x2 = min(x2_int, disp_w)
            crop_y2 = min(y2_int, disp_h)
            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                crop_img = orig_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                predicted_class = self.perform_classification(crop_img)
                text_pred = f"Cls: {predicted_class}"
                cv2.putText(
                    annotated_frame,
                    text_pred,
                    (x1_int, y2_int + text_h + baseline),
                    font,
                    font_scale,
                    (255, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

                # 누적 통계 업데이트 (track id별)
                if track_id not in self.track_classifications:
                    self.track_classifications[track_id] = predicted_class
                    self.cumulative_total += 1
                    self.cumulative_class_counts[predicted_class] += 1
                else:
                    old_class = self.track_classifications[track_id]
                    if old_class != predicted_class:
                        self.cumulative_class_counts[old_class] -= 1
                        self.cumulative_class_counts[predicted_class] += 1
                        self.track_classifications[track_id] = predicted_class

                # recent 이미지 업데이트 (크롭한 이미지 전체를 일정 사이즈로 축소)
                if predicted_class in [1, 2]:
                    crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    h_crop, w_crop, _ = crop_img_rgb.shape
                    thumb_qimg = QImage(
                        crop_img_rgb.data,
                        w_crop,
                        h_crop,
                        w_crop * 3,
                        QImage.Format_RGB888,
                    )
                    thumb_pixmap = QPixmap.fromImage(thumb_qimg)
                    target = QSize(80, 60)
                    final_thumb = create_thumbnail(thumb_pixmap, target)
                    self.recent_images_widget.update_for_class(
                        predicted_class, final_thumb
                    )
            else:
                pass

        final_display = self.update_video_display(annotated_frame)

        if valid_detection:
            self.result_label.setText("Detected")
            self.result_label.setStyleSheet("color: green;")
        else:
            self.result_label.setText("No Detection")
            self.result_label.setStyleSheet("color: green;")

        self.frame_counter += 1
        self.update_stats_display()

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
