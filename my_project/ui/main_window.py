import sys
import cv2
import numpy as np
import random
import os
import datetime
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QApplication
from PyQt5.QtCore import QDateTime, Qt, QSize, QRect, QThreadPool, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter

# 사용자 정의 위젯 및 모듈 임포트
from ui.widgets.video_label import VideoLabel
from ui.widgets.time_label import TimeLabel
from ui.widgets.result_label import ResultLabel
from ui.widgets.stats_label import StatsLabel
from ui.widgets.recent_images_widget import RecentImagesWidget
from ui.widgets.roi_label import ROILabel
from utils.sort import Sort
from model.inference_worker import InferenceWorker
from model.classification_worker import ClassificationWorker

# EfficientClassifier는 분류 모델 추론에 사용됩니다.
from model.classifier_model import EfficientClassifier


def create_thumbnail(pixmap, target_size):
    """
    원본 pixmap의 전체 비율을 유지하며 target_size(QSize)로 축소한 후,
    target_size 크기의 빈 캔버스 중앙에 배치하는 함수.
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

        # ---------------------------
        # 창 및 내부 위젯 크기 설정
        # ---------------------------
        self.window_width = 1280
        self.window_height = 720

        # 비디오 영역: 창 크기의 절반 (예: 640x360)
        self.video_width = int(self.window_width * 0.5)
        self.video_height = int(self.window_height * 0.5)

        # 최근 이미지(썸네일) 크기: 창 너비의 5%, 창 높이의 14%
        self.thumbnail_size = QSize(
            int(self.window_width * 0.05), int(self.window_height * 0.14)
        )
        # -------------------------------------------------------------------------

        self.num_classes = 3
        self.roi_threshold = 1.0

        self.tracker = Sort()
        self.track_flags = {}
        # track_classifications: 각 track의 최종 분류 결과 저장, 예) 0: 미분류, 1 혹은 2: 분류됨
        self.track_classifications = {}

        # 통계용 변수 (누적 total 건수, 클래스별 건수)
        self.cumulative_total = 0
        self.cumulative_class_counts = {i: 0 for i in range(self.num_classes)}

        # 추론 결과 및 작업자 (비동기 처리)
        self.inference_result = None
        self.inference_worker = None

        # 분류 작업 병렬 처리를 위한 QThreadPool
        self.threadpool = QThreadPool()

        import torch

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS를 사용합니다:", device)
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("CUDA GPU를 사용합니다:", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("CPU를 사용합니다.")

        # EfficientClassifier의 인스턴스 생성 시 device 전달
        self.efficient_classifier = EfficientClassifier(
            model_path="model/efficient_classifier.pth",
            num_classes=3,
            model_name="efficientnet-b0",
            device=device,
        )

        self.initialize_window()
        self.setup_ui_components()
        self.setup_webcam()
        self.setup_timers()
        self.update_time()

        # QImage 버퍼 미리 할당
        self.qimage_buffer = QImage(
            self.video_width, self.video_height, QImage.Format_RGB888
        )
        self.frame_counter = 0

    def initialize_window(self):
        self.setWindowTitle("Webcam & Inference Viewer")
        self.setGeometry(100, 100, self.window_width, self.window_height)

    def setup_ui_components(self):
        # 왼쪽 영역: 시간, 영상, 결과 표시
        self.time_label = TimeLabel()
        self.video_label = VideoLabel(width=self.video_width, height=self.video_height)
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

        # 오른쪽 영역: 통계 및 최근 이미지 표시
        self.stats_label = StatsLabel(self.num_classes)
        self.recent_images_widget = RecentImagesWidget(
            self.num_classes, thumbnail_size=self.thumbnail_size
        )
        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.stats_label)
        self.right_layout.addWidget(self.recent_images_widget)
        self.right_container = QWidget()
        self.right_container.setLayout(self.right_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.left_container)
        main_layout.addWidget(self.right_container)
        self.setLayout(main_layout)

        # 기본 ROI 설정: 비디오 영역 중앙에서 전체 크기의 60%로 설정
        default_width = int(self.video_width * 0.6)
        default_height = int(self.video_height * 0.6)
        default_x = (self.video_width - default_width) // 2
        default_y = (self.video_height - default_height) // 2
        self.roi_label.roi_rect = QRect(
            default_x, default_y, default_width, default_height
        )

    def setup_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)

    def setup_timers(self):
        self.timer_video = QTimer()
        self.timer_video.timeout.connect(self.update_frame)
        self.timer_video.start(70)  # 약 33 FPS, 추론 70ms 기준

        self.timer_time = QTimer()
        self.timer_time.timeout.connect(self.update_time)
        self.timer_time.start(1000)  # 1초마다 업데이트

    def update_time(self):
        now = QDateTime.currentDateTime().toString("yyyy/MM/dd hh:mm:ss")
        self.time_label.setText(now)

    def update_video_display(self, frame):
        if frame.shape[1] != self.video_width or frame.shape[0] != self.video_height:
            display_frame = cv2.resize(frame, (self.video_width, self.video_height))
        else:
            display_frame = frame

        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        ptr = self.qimage_buffer.bits()
        ptr.setsize(self.qimage_buffer.byteCount())
        data = np.frombuffer(ptr, dtype=np.uint8).reshape(
            (self.video_height, self.video_width, 3)
        )
        np.copyto(data, rgb_frame)

        pix = QPixmap.fromImage(self.qimage_buffer)
        self.video_label.setPixmap(pix)
        self.roi_label.setGeometry(0, 0, self.video_width, self.video_height)
        return display_frame

    def update_stats_display(self):
        self.stats_label.update_text(
            self.cumulative_total, self.cumulative_class_counts
        )

    # EfficientClassifier 모델을 이용하여 crop_img를 분류하는 함수
    def perform_classification(self, crop_img):
        try:
            predicted, confs = self.efficient_classifier.predict(crop_img)
            if predicted is None:
                print("perform_classification: predict returned None, defaulting to 0")
                predicted = 0
        except Exception as e:
            print(f"분류 작업 중 오류 발생: {e}")
            predicted = 0
        return predicted

    # classification_result 슬롯: 분류 결과를 받아 추적 결과와 통계를 업데이트
    def classification_result(self, track_id, result):
        if isinstance(result, tuple):
            predicted = result[0]
        else:
            predicted = result

        if predicted is None:
            predicted = 0

        if track_id in self.track_classifications:
            current = self.track_classifications[track_id]
            if current in [1, 2] and predicted == 0:
                return
            elif current != predicted:
                self.cumulative_class_counts[current] -= 1
                self.cumulative_class_counts[predicted] += 1
                self.track_classifications[track_id] = predicted
        else:
            self.track_classifications[track_id] = predicted
            self.cumulative_total += 1
            self.cumulative_class_counts[predicted] += 1
        print(f"Classification result for track {track_id}: {predicted}")
        self.update_stats_display()

    def handle_inference_result(self, results):
        self.inference_result = results

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        if frame.shape[1] != self.video_width or frame.shape[0] != self.video_height:
            display_frame = cv2.resize(frame, (self.video_width, self.video_height))
        else:
            display_frame = frame

        # 비동기 추론 수행
        frame_for_inference = display_frame.copy()
        if self.inference_worker is None or not self.inference_worker.isRunning():
            self.inference_worker = InferenceWorker(frame_for_inference)
            self.inference_worker.result_ready.connect(self.handle_inference_result)
            self.inference_worker.start()
        results = self.inference_result

        orig_frame = display_frame.copy()
        disp_h, disp_w, _ = display_frame.shape

        roi = self.roi_label.roi_rect
        if roi is not None:
            roi_x1 = max(0, roi.x())
            roi_y1 = max(0, roi.y())
            roi_x2 = min(disp_w, roi.x() + roi.width())
            roi_y2 = min(disp_h, roi.y() + roi.height())
        else:
            roi_x1 = roi_y1 = roi_x2 = roi_y2 = None

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

            valid_threshold_vertical = 0.8

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

                    vertical_coverage = inter_h / (y2 - y1)

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    center_inside = (
                        center_x >= roi_x1
                        and center_x <= roi_x2
                        and center_y >= roi_y1
                        and center_y <= roi_y2
                    )

                    if (
                        ratio >= self.roi_threshold
                        and vertical_coverage >= valid_threshold_vertical
                        and center_inside
                    ):
                        valid_det_info.append(
                            [x1, y1, x2, y2, confs[i], int(cls_ids[i])]
                        )
                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 0, 0),
                            1,
                        )
                else:
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
        update_text = self.frame_counter % 3 == 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        for trk in tracked_objects:
            x1, y1, x2, y2, track_id = trk
            track_id = int(track_id)
            cv2.rectangle(
                annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )

            if update_text:
                label_id = f"ID:{track_id}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_id, font, font_scale, thickness
                )
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1) - text_h - baseline),
                    (int(x1) + text_w, int(y1)),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label_id,
                    (int(x1), int(y1) - baseline),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

            crop_x1 = max(0, int(x1))
            crop_y1 = max(0, int(y1))
            crop_x2 = min(int(x2), disp_w)
            crop_y2 = min(int(y2), disp_h)
            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                crop_img = orig_frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if update_text:
                    # ClassificationWorker에 self.efficient_classifier 인스턴스를 전달합니다.
                    worker = ClassificationWorker(
                        crop_img, track_id, self.efficient_classifier.predict
                    )
                    worker.signals.result_ready.connect(self.classification_result)
                    self.threadpool.start(worker)
                predicted_class = self.track_classifications.get(track_id, None)
                if predicted_class is not None and update_text:
                    text_pred = f"Cls: {predicted_class}"
                    cv2.putText(
                        annotated_frame,
                        text_pred,
                        (int(x1), int(y2) + text_h + baseline),
                        font,
                        font_scale,
                        (255, 0, 0),
                        thickness,
                        cv2.LINE_AA,
                    )

                # 최근 이미지(썸네일) 업데이트: 분류 결과가 1 또는 2이면 항상 업데이트
                if (
                    predicted_class in [1, 2]
                    and crop_x2 > crop_x1
                    and crop_y2 > crop_y1
                ):
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
                    self.recent_images_widget.update_for_class(
                        predicted_class, thumb_pixmap
                    )

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
