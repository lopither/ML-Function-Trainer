import sys

from PyQt6.QtWidgets import QApplication

from app.gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ML Function Visualizer")
    app.setOrganizationName("ML Function Visualizer")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
