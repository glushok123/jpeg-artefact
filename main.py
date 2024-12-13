import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QProgressBar,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QHBoxLayout,
    QMessageBox, QHeaderView, QSlider, QDialog, QAbstractItemView
)
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, UnidentifiedImageError
import subprocess


def is_jpeg(file_path):
    """Check if the file is a JPEG by signature."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(2)
        return header == b'\xff\xd8'
    except Exception:
        return False


def check_jpeg_integrity(file_path):
    """Verify the integrity of a JPEG file."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, IOError):
        return False


# Фильтрация и анализ градиента
def detect_with_gradient(image, threshold):
    """Проверка артефактов с использованием градиента (Собеля)."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_horizontal = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        max_gradient = np.max(np.abs(sobel_horizontal))
        return max_gradient > threshold, {"max_gradient": max_gradient}
    except Exception as e:
        return False, {"error": str(e)}


# Частотный анализ (FFT)
def detect_with_fft(image, threshold):
    """Проверка артефактов с использованием FFT."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))
        max_frequency = np.max(magnitude_spectrum)
        return max_frequency > threshold, {"max_frequency": max_frequency}
    except Exception as e:
        return False, {"error": str(e)}


# Локальная энтропия
def detect_with_entropy(image, threshold):
    """Проверка артефактов с использованием локальной энтропии."""
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entropy_image = entropy(gray, disk(5))
        max_entropy = np.max(entropy_image)
        return max_entropy > threshold, {"max_entropy": max_entropy}
    except Exception as e:
        return False, {"error": str(e)}


# Полосатость
def detect_with_stripiness(image, threshold):
    """Проверка артефактов полосатости."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        row_means = np.mean(gray, axis=1)
        std_dev = np.std(row_means)
        return std_dev > threshold, {"std_dev": std_dev}
    except Exception as e:
        return False, {"error": str(e)}


def has_black_border(image, threshold, border_info):
    """Check if the image has a black border on top-bottom or left-right sides."""
    try:
        # Считаем средние значения пикселей для краев
        top_border = np.mean(image[0, :])
        bottom_border = np.mean(image[-1, :])
        left_border = np.mean(image[:, 0])
        right_border = np.mean(image[:, -1])

        # Проверяем пары краев
        top_bottom_black = top_border < threshold and bottom_border < threshold
        left_right_black = left_border < threshold and right_border < threshold

        # Сохраняем значения в словарь
        border_info["top_border"] = top_border
        border_info["bottom_border"] = bottom_border
        border_info["left_border"] = left_border
        border_info["right_border"] = right_border

        if top_bottom_black:
            return True

        # if top_border + bottom_border < 170 and left_border + right_border < 170:
        #    return True

        if (top_border > 100 or bottom_border > 100 or left_border > 100 or right_border > 100):
            return False
        # Возвращаем True, если хотя бы одна пара краев имеет черные рамки
        return top_bottom_black or left_right_black
    except Exception:
        return False


def detect_horizontal_artifacts(file_path, threshold_difference, border_threshold):
    """
    Проверка на наличие горизонтальных артефактов с использованием разных методов.
    """
    results = {}
    try:
        image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return False, "Failed to read image.", results

        # Проверка черных рамок
        black_border_info = {}
        if has_black_border(image, border_threshold, black_border_info):
            return False, "Изображение исправно!", black_border_info

        results = black_border_info
        if(results['bottom_border'] > 125):
            return True, "artifact", results
        height, width, _ = image.shape
        cropped_image = image[200:height - 200, 200:width - 200]

        image = cropped_image
        # Метод 1: Разница между блоками
        detected_dif = False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        block_size = 300
        height, _ = gray.shape
        num_blocks = height // block_size
        maxDif = 0
        for i in range(num_blocks - 1):
            block1 = gray[i * block_size:(i + 1) * block_size, :]
            block2 = gray[(i + 1) * block_size:(i + 2) * block_size, :]
            mean1 = np.mean(block1)
            mean2 = np.mean(block2)
            diff = abs(mean2 - mean1)
            results["Разница"] = diff
            if (diff > maxDif):
                maxDif = diff
                results["Разница"] = maxDif
            if diff > threshold_difference:
                results["Разница"] = diff
                detected_dif = True
                break
        else:
            results["difference_blocks"] = "No significant difference detected"

        # Метод 2: Градиенты
        # detected_grad, grad_info = detect_with_gradient(image, 100)
        # results["gradient_detection"] = grad_info if detected_grad else "No gradient artifacts"

        # Метод 3: FFT
        detected_fft, fft_info = detect_with_fft(image, 1000)
        results["fft_detection"] = fft_info if detected_fft else "No frequency artifacts"

        # Метод 4: Локальная энтропия
        # detected_entropy, entropy_info = detect_with_entropy(image, 5)
        # results["entropy_detection"] = entropy_info if detected_entropy else "No entropy anomalies"

        # Метод 5: Полосатость
        detected_stripiness, stripiness_info = detect_with_stripiness(image, 15)
        results["stripiness_detection"] = stripiness_info if detected_stripiness else "No stripiness detected"

        if detected_stripiness == False and detected_fft == False and detected_dif == False:
            return False, "No horizontal artifacts detected.", {}

        return True, "artifact", results
    except Exception as e:
        return False, f"Error: {str(e)}", results


class WorkerSignals(QObject):
    progress_signal = pyqtSignal(dict)
    result_signal = pyqtSignal(str, str, object)


class FileProcessor(QRunnable):
    def __init__(self, file_path, threshold_difference, border_threshold):
        super().__init__()
        self.file_path = file_path
        self.threshold_difference = threshold_difference
        self.border_threshold = border_threshold
        self.signals = WorkerSignals()

    def run(self):
        integrity_ok = check_jpeg_integrity(self.file_path)
        if not integrity_ok:
            result = "Файл поврежден."
            self.signals.result_signal.emit(self.file_path, result, {})
            self.signals.progress_signal.emit({'processed': 1, 'corrupt': 1, 'artifacts': 0})
        else:
            artifacts_detected, message, info = detect_horizontal_artifacts(
                self.file_path, self.threshold_difference, self.border_threshold)
            if artifacts_detected:
                result = "Обнаружены артефакты."
                self.signals.result_signal.emit(self.file_path, result, info)
                self.signals.progress_signal.emit({'processed': 1, 'corrupt': 0, 'artifacts': 1})
            else:
                self.signals.progress_signal.emit({'processed': 1, 'corrupt': 0, 'artifacts': 0})


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Проверка JPEG файлов")
        self.setGeometry(100, 100, 1300, 800)

        self.files = []
        self.threadpool = QThreadPool()
        self.stats = {'processed': 0, 'corrupt': 0, 'artifacts': 0}
        self.start_time = None

        # Значения, регулируемые через интерфейс
        self.threshold_difference = 45  # Для горизонтальных артефактов
        self.black_border_threshold = 56  # Пороговое значение для черной рамки

        self.init_ui()

    def init_ui(self):
        # Layouts
        layout = QVBoxLayout()

        # Select Directory Button
        self.select_button = QPushButton("Выбрать папку")
        self.select_button.clicked.connect(self.select_directory)
        layout.addWidget(self.select_button)

        # Пороговая разница
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Пороговая разница:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(self.threshold_difference)
        self.threshold_slider.setTickInterval(5)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)

        self.threshold_value_label = QLabel(str(self.threshold_difference))
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_value_label)
        layout.addLayout(threshold_layout)

        # Пороговое значение для черной рамки
        border_threshold_layout = QHBoxLayout()
        border_threshold_label = QLabel("Порог черной рамки:")
        self.border_threshold_slider = QSlider(Qt.Horizontal)
        self.border_threshold_slider.setMinimum(10)
        self.border_threshold_slider.setMaximum(100)
        self.border_threshold_slider.setValue(self.black_border_threshold)
        self.border_threshold_slider.setTickInterval(5)
        self.border_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.border_threshold_slider.valueChanged.connect(self.update_border_threshold_label)

        self.border_threshold_value_label = QLabel(str(self.black_border_threshold))
        border_threshold_layout.addWidget(border_threshold_label)
        border_threshold_layout.addWidget(self.border_threshold_slider)
        border_threshold_layout.addWidget(self.border_threshold_value_label)
        layout.addLayout(border_threshold_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Statistics Label
        self.stats_label = QLabel("Обработано: 0 | Повреждено: 0 | С артефактами: 0 | Время: 0с")
        layout.addWidget(self.stats_label)

        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Путь к файлу", "Статус", "Миниатюра", "Действия", "Флаг", "Информация"
        ])

        # --- Настройки прокрутки ---
        self.results_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.results_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.results_table.verticalScrollBar().setSingleStep(20)  # Шаг прокрутки
        self.results_table.horizontalScrollBar().setSingleStep(1)  # Для горизонтальной прокрутки

        # Настройки стилей скроллбара (опционально)
        self.results_table.verticalScrollBar().setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background: #f5f5f5;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #3EB6FA;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical {
                height: 0px;
                subcontrol-position: bottom;
            }
            QScrollBar::sub-line:vertical {
                height: 0px;
                subcontrol-position: top;
            }
        """)
        # --------------------------

        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSortingEnabled(True)
        self.results_table.setColumnHidden(4, True)
        layout.addWidget(self.results_table)

        # Control Buttons
        control_layout = QHBoxLayout()

        self.start_button = QPushButton("Запустить")
        self.start_button.clicked.connect(self.start_analysis)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Остановить")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        layout.addLayout(control_layout)

        # Main Widget
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def update_threshold_label(self):
        """Обновить значение пороговой разницы в метке."""
        self.threshold_difference = self.threshold_slider.value()
        self.threshold_value_label.setText(str(self.threshold_difference))

    def update_border_threshold_label(self):
        """Обновить значение порогового значения черной рамки в метке."""
        self.black_border_threshold = self.border_threshold_slider.value()
        self.border_threshold_value_label.setText(str(self.black_border_threshold))

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if directory:
            self.files = [
                os.path.join(root, file)
                for root, _, files in os.walk(directory)
                for file in files
                if file.lower().endswith(('.jpg', '.jpeg')) or is_jpeg(os.path.join(root, file))
            ]
            self.stats_label.setText(f"Найдено {len(self.files)} JPEG файлов.")

    def start_analysis(self):
        if not self.files:
            QMessageBox.warning(self, "Нет файлов", "Выберите папку с JPEG файлами.")
            return

        self.results_table.setRowCount(0)
        self.progress_bar.setMaximum(len(self.files))
        self.progress_bar.setValue(0)
        self.start_time = time.time()
        self.stats = {'processed': 0, 'corrupt': 0, 'artifacts': 0}
        self.stats_label.setText("Обработано: 0 | Повреждено: 0 | С артефактами: 0 | Время: 0с")

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.threadpool.setMaxThreadCount(os.cpu_count() or 1)

        self.tasks = []
        for file_path in self.files:
            worker = FileProcessor(file_path, self.threshold_difference, self.black_border_threshold)
            worker.signals.progress_signal.connect(self.update_progress)
            worker.signals.result_signal.connect(self.show_result)
            self.threadpool.start(worker)
            self.tasks.append(worker)

    def stop_analysis(self):
        self.threadpool.clear()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        elapsed_time = int(time.time() - self.start_time) if self.start_time else 0
        self.stats_label.setText(f"Статус: Остановлено. Время: {elapsed_time}с")

    def update_progress(self, stats):
        self.stats['processed'] += stats['processed']
        self.stats['corrupt'] += stats['corrupt']
        self.stats['artifacts'] += stats['artifacts']

        elapsed_time = int(time.time() - self.start_time) if self.start_time else 0
        self.progress_bar.setValue(self.stats['processed'])
        self.stats_label.setText(
            f"Обработано: {self.stats['processed']} | Повреждено: {self.stats['corrupt']} | С артефактами: {self.stats['artifacts']} | Время: {elapsed_time}с"
        )

        if self.stats['processed'] >= len(self.files):
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def show_result(self, file_path, result, info):
        row_position = self.results_table.rowCount()
        self.results_table.insertRow(row_position)

        self.results_table.setItem(row_position, 0, QTableWidgetItem(file_path))
        self.results_table.setItem(row_position, 1, QTableWidgetItem(result))

        # Если файл имеет артефакты, создаем миниатюру
        if result == "Обнаружены артефакты.":
            thumbnail_label = self.create_thumbnail_label(file_path)
            self.results_table.setCellWidget(row_position, 2, thumbnail_label)
            self.results_table.setItem(row_position, 4, QTableWidgetItem("1"))  # Флаг наличия миниатюры
            self.results_table.setItem(row_position, 5, QTableWidgetItem(self.format_info(info)))
        else:
            self.results_table.setItem(row_position, 2, QTableWidgetItem(""))
            self.results_table.setItem(row_position, 4, QTableWidgetItem("0"))  # Флаг отсутствия миниатюры

        open_explorer_button = QPushButton("Открыть в проводнике")
        open_explorer_button.setFixedSize(140, 20)
        open_explorer_button.clicked.connect(lambda _, fp=file_path: self.open_in_explorer(fp))

        view_image_button = QPushButton("Открыть изображение")
        view_image_button.setFixedSize(140, 20)
        view_image_button.clicked.connect(lambda _, fp=file_path: self.view_image(fp))

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(open_explorer_button)
        actions_layout.addWidget(view_image_button)
        actions_layout.setContentsMargins(0, 0, 0, 0)

        actions_widget = QWidget()
        actions_widget.setLayout(actions_layout)
        self.results_table.setCellWidget(row_position, 3, actions_widget)

        # Устанавливаем высоту строки по содержимому
        self.results_table.resizeRowToContents(row_position)

    def format_info(self, info):
        """Форматирует словарь в строку, где каждое значение начинается с новой строки."""
        lines = []
        for key, value in info.items():
            if isinstance(value, dict):
                # Форматируем вложенные словари
                nested = "\n".join([f"  {nested_key}: {nested_value}" for nested_key, nested_value in value.items()])
                lines.append(f"{key}:\n{nested}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def create_thumbnail_label(self, file_path):
        try:
            # Создаем миниатюру
            img = Image.open(file_path)
            img.thumbnail((250, 250), Image.Resampling.LANCZOS)  # Используем LANCZOS вместо ANTIALIAS

            # Преобразуем в QPixmap
            img_data = img.convert("RGBA")
            data = img_data.tobytes("raw", "RGBA")
            qimage = QImage(data, img_data.width, img_data.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)

            # Создаем QLabel с миниатюрой
            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)

            # Добавляем обработку клика
            label.mousePressEvent = lambda event, path=file_path: self.show_full_image(path)

            return label
        except Exception as e:
            print(f"Ошибка при создании миниатюры для {file_path}: {e}")
            label = QLabel("Ошибка")
            label.setAlignment(Qt.AlignCenter)
            return label

    def show_full_image(self, file_path):
        """Открыть изображение на всю высоту экрана."""
        try:
            # Загружаем изображение
            img = Image.open(file_path)
            img = img.convert("RGBA")

            # Преобразуем в QPixmap
            data = img.tobytes("raw", "RGBA")
            qimage = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)

            # Создаем QDialog для отображения изображения
            dialog = QDialog(self)
            dialog.setWindowTitle("Просмотр изображения")
            dialog.setModal(True)

            # Устанавливаем размеры окна в зависимости от экрана
            screen_height = QApplication.primaryScreen().geometry().height()
            scale_factor = screen_height / img.height
            scaled_pixmap = pixmap.scaled(
                int(img.width * scale_factor),  # Преобразуем в целое число
                int(screen_height),  # Преобразуем в целое число
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Создаем QLabel для изображения
            label = QLabel(dialog)
            label.setPixmap(scaled_pixmap)
            label.setAlignment(Qt.AlignCenter)

            # Настройка диалогового окна
            layout = QVBoxLayout(dialog)
            layout.addWidget(label)
            dialog.setLayout(layout)
            dialog.resize(scaled_pixmap.width(), screen_height)

            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть изображение: {e}")

    def open_in_explorer(self, file_path):
        try:
            # Открываем проводник с выделенным файлом
            subprocess.run(["explorer", "/select,", os.path.normpath(file_path)])
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось открыть проводник: {e}")

    def view_image(self, file_path):
        try:
            # Убедимся, что путь корректный
            file_path = os.path.normpath(file_path)
            img = Image.open(file_path)
            img.show()
        except UnidentifiedImageError:
            QMessageBox.critical(self, "Ошибка",
                                 "Не удалось открыть изображение. Файл поврежден или не является изображением.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при открытии изображения: {e}")

    def closeEvent(self, event):
        self.threadpool.waitForDone()
        event.accept()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
