from PyQt5.QtCore import QRunnable, pyqtSignal, QObject


# SignalEmitter 클래스는 작업 완료 후 결과를 메인 스레드로 전달하기 위한 도우미 클래스입니다.
class SignalEmitter(QObject):
    # 첫 번째 인자는 track_id, 두 번째 인자는 예측 결과(예: (predicted, probabilities))
    result_ready = pyqtSignal(int, object)


# ClassificationWorker 클래스는 전달받은 classifier(예: EfficientClassifier 인스턴스)의 predict() 메서드를 이용하여
# crop_img의 분류를 비동기로 수행하고, 그 결과를 SignalEmitter를 통해 전달합니다.
class ClassificationWorker(QRunnable):
    def __init__(self, crop_img, track_id, classifier):
        """
        crop_img: 분류할 이미지 영역 (BGR numpy 배열 등, classifier.predict()가 처리할 수 있는 형식)
        track_id: 작업별 식별자 (예: 추적 ID 등)
        classifier: 분류 기능을 수행하는 객체로, predict(input_image) 메서드를 제공해야 합니다.
        """
        super().__init__()
        self.crop_img = crop_img
        self.track_id = track_id
        self.classifier = classifier
        self.signals = SignalEmitter()  # 결과 전달용 SignalEmitter

    def run(self):
        try:
            # EfficientClassifier나 다른 분류 클래스의 predict() 메서드를 호출합니다.
            # predict 메서드는 (predicted, probabilities)를 반환한다고 가정합니다.
            predicted, probs = self.classifier.predict(self.crop_img)
        except Exception as e:
            print(f"분류 작업 중 오류 발생 (track_id: {self.track_id}):", e)
            predicted, probs = None, None
        # 작업 결과를 메인 스레드에 전달 (예: track_id와 결과 튜플)
        self.signals.result_ready.emit(self.track_id, (predicted, probs))
