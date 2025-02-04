from PyQt5.QtCore import QRunnable, pyqtSignal, QObject


# SignalEmitter는 작업 완료 시 메인 스레드에 결과를 전달하기 위한 도우미 클래스입니다.
class SignalEmitter(QObject):
    result_ready = pyqtSignal(int, object)  # track_id와 예측 결과를 전달


class ClassificationWorker(QRunnable):
    def __init__(self, crop_img, track_id, perform_classification_func):
        super().__init__()
        self.crop_img = crop_img
        self.track_id = track_id
        self.perform_classification = perform_classification_func
        self.signals = SignalEmitter()  # 결과 신호

    def run(self):
        # 이곳에서 분류 작업을 수행합니다.
        predicted = self.perform_classification(self.crop_img)
        # self.signals.result_ready를 통해 결과 전송
        self.signals.result_ready.emit(self.track_id, predicted)
