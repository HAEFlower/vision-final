import sys
import cv2
import numpy as np
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
from utils.sort import Sort

# 사전 로드된 YOLO 모델
yolo_model = load_model()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # 예시: 클래스 수를 3으로 설정 (필요에 따라 수정)
        self.num_classes = 3
        # ROI 영역 내에 객체가 차지해야 하는 최소 비율 (0.6 = 60%)
        self.roi_threshold = 0.6

        self.tracker = Sort()
        self.track_flags = {}
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

        # 좌우 미러링 및 640x360 리사이즈
        frame = cv2.flip(frame, 1)
        display_frame = cv2.resize(frame, (640, 360))
        disp_h, disp_w, _ = display_frame.shape

        # ROI 영역 (ROILabel에서 지정; display_frame 좌표 기준)
        roi = self.roi_label.roi_rect
        if roi is not None:
            roi_x1 = max(0, roi.x())
            roi_y1 = max(0, roi.y())
            roi_x2 = min(disp_w, roi.x() + roi.width())
            roi_y2 = min(disp_h, roi.y() + roi.height())

        # YOLO inference (전체 화면 기준)
        try:
            results = run_inference(yolo_model, display_frame)
        except Exception as e:
            print("YOLO detection error:", e)
            results = None

        # valid_det_info: [x1, y1, x2, y2, conf, class_id] for detections passing ROI threshold
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
                # 기준 비율 통과라면 유효 검출
                if ratio >= self.roi_threshold:
                    valid_det_info.append([x1, y1, x2, y2, confs[i], int(cls_ids[i])])
                    # (옵션) 원본에 파란색 박스 그리기
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0),
                        1,
                    )

        # SORT tracker 업데이트 — tracker.update()는 [x1,y1,x2,y2,conf] 형식만 필요
        if len(valid_det_info) > 0:
            dets = np.array(valid_det_info)[:, :5]  # shape (N, 5)
            tracked_objects = self.tracker.update(dets)
        else:
            tracked_objects = np.empty((0, 5))

        valid_detection = False
        current_track_ids = set()

        # 각 트랙에 대해 persistent classification 수행
        for trk in tracked_objects:
            x1, y1, x2, y2, track_id = trk
            track_id = int(track_id)
            current_track_ids.add(track_id)
            # 매 트랙에 대해 가장 좋은 매칭 검출을 찾기 위해 IoU 계산
            best_iou = 0
            best_det = None
            for det in valid_det_info:
                iou_val = self.compute_iou([x1, y1, x2, y2], det[:4])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = det
            if best_iou > 0.5 and best_det is not None:
                det_class = best_det[5]
                # 만약 검출된 클래스가 0이 아니라면 persistent flag를 True
                if det_class != 0:
                    self.track_flags[track_id] = True
                else:
                    if track_id not in self.track_flags:
                        self.track_flags[track_id] = False
            else:
                if track_id not in self.track_flags:
                    self.track_flags[track_id] = False

            # 트랙이 persistent flag가 True이면 valid_detection으로 판단
            if self.track_flags.get(track_id, False):
                valid_detection = True

            # 화면에 tracked box와 id 표시 (초록색)
            cv2.rectangle(
                annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
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

        # 트랙 ID 업데이트 후, dictionary에서 더 이상 보이지 않는 ID 제거
        remove_keys = [tid for tid in self.track_flags if tid not in current_track_ids]
        for tid in remove_keys:
            del self.track_flags[tid]

        final_display = self.update_video_display(annotated_frame)

        # 결과 라벨 업데이트
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

    def compute_iou(self, boxA, boxB):
        # box format: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
