from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal


class SnapshotAnimator(QObject):
    """Timer-based playback controller for stored prediction snapshots."""

    frame_changed = pyqtSignal(int, object)
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._snapshots: list[object] = []
        self._index = 0
        self._interval_ms = 200
        self._timer.setInterval(self._interval_ms)

    @property
    def current_index(self) -> int:
        return self._index

    def set_snapshots(self, snapshots: list[object]) -> None:
        self.pause()
        self._snapshots = list(snapshots)
        self._index = 0

    def has_snapshots(self) -> bool:
        return bool(self._snapshots)

    def set_speed(self, speed_value: int) -> None:
        speed = max(1, int(speed_value))
        self._interval_ms = max(30, int(1000 / speed))
        self._timer.setInterval(self._interval_ms)

    def play(self) -> None:
        if not self._snapshots:
            return
        if self._index >= len(self._snapshots) - 1:
            self._index = 0
            self.frame_changed.emit(self._index, self._snapshots[self._index])
        self._timer.start()
        self.playback_started.emit()

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.playback_stopped.emit()

    def reset(self) -> None:
        self.pause()
        self._index = 0
        if self._snapshots:
            self.frame_changed.emit(self._index, self._snapshots[self._index])

    def step_forward(self) -> None:
        if not self._snapshots:
            return
        self.pause()
        self._index = min(self._index + 1, len(self._snapshots) - 1)
        self.frame_changed.emit(self._index, self._snapshots[self._index])

    def step_backward(self) -> None:
        if not self._snapshots:
            return
        self.pause()
        self._index = max(self._index - 1, 0)
        self.frame_changed.emit(self._index, self._snapshots[self._index])

    def _advance(self) -> None:
        if not self._snapshots:
            self.pause()
            return

        if self._index >= len(self._snapshots) - 1:
            self.pause()
            return

        self._index += 1
        self.frame_changed.emit(self._index, self._snapshots[self._index])
