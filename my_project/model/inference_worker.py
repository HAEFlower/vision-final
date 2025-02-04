from PyQt5.QtCore import QThread, pyqtSignal
from model.yolo_model import run_inference, load_model

# 추론에 사용될 YOLO 모델을 미리 로드
yolo_model = load_model()


class InferenceWorker(QThread):
    result_ready = pyqtSignal(object)  # 추론 결과를 emit

    def __init__(self, frame, parent=None):
        super().__init__(parent)
        self.frame = frame

    def run(self):
        try:
            results = run_inference(yolo_model, self.frame)
        except Exception as e:
            print("Inference error:", e)
            results = None
        self.result_ready.emit(results)
