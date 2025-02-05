from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt


class StatsLabel(QLabel):
    def __init__(self, num_classes, parent=None):
        super().__init__(parent)
        self.num_classes = num_classes
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setTextFormat(Qt.RichText)
        # 클래스 번호에 대한 레이블 매핑
        self.labels = {0: "NG1", 1: "NG2", 2: "GOOD"}
        self.setText(self.build_initial_text())

    def build_initial_text(self):
        stats_html = (
            "<p style='font-size:30pt; font-weight:bold; margin:0 0 20px;'>Total: 0</p>"
        )
        for cls in range(self.num_classes):
            label = self.labels.get(cls, f"Class {cls}")
            stats_html += (
                f"<p style='font-size:24pt; margin:0 0 15px;'>{label}: 0 (0.0%)</p>"
            )
        return stats_html

    def update_text(self, total_count, class_counts):
        stats_html = f"<p style='font-size:30pt; font-weight:bold; margin:0 0 20px;'>Total: {total_count}</p>"
        for cls in range(self.num_classes):
            count = class_counts.get(cls, 0)
            percent = (count / total_count * 100) if total_count > 0 else 0.0
            label = self.labels.get(cls, f"Class {cls}")
            stats_html += f"<p style='font-size:24pt; margin:0 0 15px;'>{label}: {count} ({percent:.1f}%)</p>"
        self.setText(stats_html)
