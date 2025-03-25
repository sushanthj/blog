import sys
import subprocess
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.jekyll_process = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Jekyll Server Control")
        self.setGeometry(100, 100, 300, 100)

        self.start_button = QPushButton("Start Server", self)
        self.start_button.clicked.connect(self.start_server)

        self.stop_button = QPushButton("Stop Server", self)
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setEnabled(False)

        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def start_server(self):
        if self.jekyll_process and self.jekyll_process.poll() is None:
            QMessageBox.warning(self, "Warning", "Server is already running.")
            return

        try:
            # Use Popen for non-blocking execution
            self.jekyll_process = subprocess.Popen(["/app/serve.sh"])
            QMessageBox.information(self, "Success", "Jekyll server started successfully!")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "serve.sh not found at /app/serve.sh!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
            self.jekyll_process = None

    def stop_server(self):
        if not (self.jekyll_process and self.jekyll_process.poll() is None):
            QMessageBox.warning(self, "Warning", "Jekyll server is not running.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.jekyll_process = None
            return

        try:
            self.jekyll_process.terminate()
            self.jekyll_process.wait(timeout=5)  # Wait for graceful shutdown
            QMessageBox.information(self, "Success", "Jekyll server stopped.")
        except subprocess.TimeoutExpired:
            self.jekyll_process.kill()  # Force kill if it doesn't respond
            QMessageBox.warning(self, "Warning", "Server did not stop gracefully and was killed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop Jekyll server: {e}")
        finally:
            self.jekyll_process = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        """Ensure the server is stopped when the window is closed."""
        if self.jekyll_process and self.jekyll_process.poll() is None:
            self.stop_server()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())