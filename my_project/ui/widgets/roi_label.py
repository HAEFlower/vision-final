from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen


class ROILabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 마우스 추적 활성화
        self.setMouseTracking(True)
        # ROI 동작 관련 변수 초기화
        self.drawing_roi = False
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None  # 최종 지정된 ROI 영역(QRect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # 왼쪽 버튼 클릭 시, ROI 그리기를 시작
            self.drawing_roi = True
            self.roi_start = event.pos()
            self.roi_end = event.pos()
            self.roi_rect = None  # 기존 ROI 초기화
            self.update()
        elif event.button() == Qt.RightButton:
            # 오른쪽 버튼 클릭 시, ROI를 초기화(삭제)하고 그리기 모드 종료
            self.drawing_roi = False
            self.roi_start = None
            self.roi_end = None
            self.roi_rect = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_roi:
            self.roi_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing_roi:
            self.roi_end = event.pos()
            self.roi_rect = self.makeRect(self.roi_start, self.roi_end)
            self.drawing_roi = False
            self.update()

    def paintEvent(self, event):
        # 기본 QLabel 그리기
        super().paintEvent(event)
        # ROI 그리기: 현재 드래그 중이거나 완료된 ROI가 있으면 그리기
        if (self.drawing_roi and self.roi_start and self.roi_end) or self.roi_rect:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.DashLine)
            painter.setPen(pen)
            if self.drawing_roi:
                rect = self.makeRect(self.roi_start, self.roi_end)
                painter.drawRect(rect)
            elif self.roi_rect:
                painter.drawRect(self.roi_rect)

    def makeRect(self, start, end):
        """
        start와 end 점으로부터 ROI 사각형을 생성하고,
        위젯 전체 영역(self.rect())과의 교차 영역을 반환하여
        ROI가 위젯 범위를 넘지 않도록 합니다.
        """
        rawRect = QRect(start, end).normalized()
        clippedRect = rawRect.intersected(self.rect())
        return clippedRect
