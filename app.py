# cwod_gui.py
import os
import sys
import time
import csv
from datetime import datetime

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QMessageBox,
    QButtonGroup, QGraphicsScene, QTableWidgetItem
)
from PySide6.QtWidgets import QHeaderView, QAbstractItemView

from ui_paper import Ui_Form


# -------------------------

# -------------------------
def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


# -------------------------

# -------------------------
class ModelLoader(QThread):
    loaded = Signal(object, str)  # (model, path)
    failed = Signal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path

    def run(self):
        try:
            from ultralytics_local import YOLO
            if not os.path.exists(self.model_path):
                self.failed.emit(f"Model not found: {self.model_path}")
                return
            model = YOLO(self.model_path)
            self.loaded.emit(model, self.model_path)
        except Exception as e:
            self.failed.emit(str(e))


# -------------------------

# -------------------------
class StreamWorker(QThread):
    frame_out = Signal(QImage)
    fps_out = Signal(float)
    det_out = Signal(list)
    info_out = Signal(str)

    def __init__(self, source, is_camera: bool):
        super().__init__()
        self.source = source
        self.is_camera = is_camera
        self._running = True

        self.model = None
        self.detect_on = False
        self.conf = 0.25
        self.iou = 0.45
        self.model_name = ""

        self._last_table_t = 0.0

    def update_model(self, model, model_name: str):
        self.model = model
        self.model_name = model_name

    def update_params(self, conf: float, iou: float):
        self.conf = conf
        self.iou = iou

    def set_detect(self, on: bool):
        self.detect_on = on

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(0 if self.is_camera else self.source)
        if not cap.isOpened():
            self.info_out.emit("Failed to open video/camera")
            return

        t_prev = time.perf_counter()
        fps_smooth = 0.0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.info_out.emit("Video ended / read failure")
                break

            det_rows = []
            show_frame = frame

            if self.detect_on:
                if self.model is None:
                    self.info_out.emit("No model loaded. Detection disabled.")
                else:
                    try:
                        res = self.model.predict(
                            frame, conf=self.conf, iou=self.iou, verbose=False
                        )[0]
                        show_frame = res.plot()

                        if res.boxes is not None and len(res.boxes) > 0:
                            xyxy = res.boxes.xyxy.cpu().numpy()
                            confs = res.boxes.conf.cpu().numpy()
                            clss = res.boxes.cls.cpu().numpy().astype(int)
                            names = res.names

                            for i in range(len(clss)):
                                x1, y1, x2, y2 = xyxy[i].tolist()
                                w = max(0.0, x2 - x1)
                                h = max(0.0, y2 - y1)

                                det_rows.append({
                                    "no": i + 1,
                                    "cls": names.get(int(clss[i]), str(int(clss[i]))),
                                    "conf": float(confs[i]),
                                    "x": float(x1),
                                    "y": float(y1),
                                    "w": float(w),
                                    "h": float(h),
                                    "model": self.model_name,
                                    "src": "stream",
                                })
                    except Exception as e:
                        self.info_out.emit(f"Inference error: {e}")

            self.frame_out.emit(bgr_to_qimage(show_frame))

            t_now = time.perf_counter()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            fps_smooth = inst_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * inst_fps)
            self.fps_out.emit(fps_smooth)

            if (t_now - self._last_table_t) > 0.1:
                self.det_out.emit(det_rows)
                self._last_table_t = t_now

        cap.release()


# -------------------------

# -------------------------
class App(QWidget):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATHS = {
        'Foggy': os.path.join(BASE_DIR, 'CWOD_fog.pt'),
        'Rainy': os.path.join(BASE_DIR, 'CWOD_rain.pt'),
        'Dark': os.path.join(BASE_DIR, 'CWOD_dark.pt'),
    }

    SELECT_GREEN = "#6FBF73"

    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.setWindowTitle("CWOD Detection System")

        self.current_source_type = None
        self.current_image_path = None
        self.current_video_path = None

        self.model = None
        self.model_path = ""
        self.model_key = "Rainy"
        self.detect_on = False

        self.stream_worker = None

        self.last_qimg = None
        self.last_rows = []
        self.last_source_name = ""

        self._setup_view()
        self._setup_table()
        self._setup_sliders()
        self._setup_weather_buttons()
        self._bind_buttons()
        self._setup_info_labels()

        self._select_weather("Rainy", auto_load=True)

    def _setup_info_labels(self):
        self.ui.label_FPS.setText("FPS: --")
        self.ui.label_Total.setText("Total: 0")

        if hasattr(self.ui, 'label_Status'):
            self.ui.label_Status.setText('Status: Ready')

    def _set_status(self, text: str):
        if hasattr(self.ui, 'label_Status') and self.ui.label_Status is not None:
            if str(text).startswith('Status:'):
                self.ui.label_Status.setText(str(text))
            else:
                self.ui.label_Status.setText(f"Status: {text}")

    def _on_stream_info(self, s: str):
        print('[INFO]', s)
        self._set_status(s)

    def _setup_view(self):
        self.scene = QGraphicsScene(self)
        self.ui.graphicsView.setScene(self.scene)

    def _setup_table(self):
        headers = ["Class", "Conf", "X", "Y", "W", "H"]
        t = self.ui.tableWidget

        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(0)

        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.verticalHeader().setVisible(False)

        t.setAlternatingRowColors(True)
        t.setShowGrid(False)
        t.verticalHeader().setDefaultSectionSize(30)

        header = t.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)

        header.setSectionResizeMode(0, QHeaderView.Fixed)
        t.setColumnWidth(0, 120)

        for col in range(1, 6):
            header.setSectionResizeMode(col, QHeaderView.Stretch)

        t.setStyleSheet("""
        QTableWidget {
            background: #FFFFFF;
            alternate-background-color: #FAFAFA;
            border: 1px solid #E6E6E6;
            border-radius: 8px;
            selection-background-color: #DFF0E2;
            selection-color: #111;
        }
        QTableWidget::item {
            padding: 4px 10px;   /* keep moderate padding */
            border: none;
        }
        QHeaderView::section {
            background: #F4F6F8;
            color: #222;
            padding: 8px 10px;
            border: none;
            font-weight: 600;
        }
        """)

    def _setup_sliders(self):
        self.ui.horizontalSlider_Confidence.setRange(1, 99)
        self.ui.horizontalSlider_IoU.setRange(1, 99)
        self.ui.horizontalSlider_Confidence.setValue(25)
        self.ui.horizontalSlider_IoU.setValue(45)

        self.ui.horizontalSlider_Confidence.valueChanged.connect(self._on_conf_changed)
        self.ui.horizontalSlider_IoU.valueChanged.connect(self._on_iou_changed)

        self._on_conf_changed(self.ui.horizontalSlider_Confidence.value())
        self._on_iou_changed(self.ui.horizontalSlider_IoU.value())

    def _apply_weather_style(self, btn, base_color: str):
        btn.setStyleSheet(f"""
        QToolButton {{
            background-color: {base_color};
            border-radius: 10px;
        }}
        QToolButton:checked {{
            background-color: {self.SELECT_GREEN};
        }}
        """)

    def _setup_weather_buttons(self):
        for b in (self.ui.toolButton_Rainy, self.ui.toolButton_Foggy, self.ui.toolButton_Dark):
            b.setCheckable(True)

        self.weather_group = QButtonGroup(self)
        self.weather_group.setExclusive(True)
        self.weather_group.addButton(self.ui.toolButton_Rainy)
        self.weather_group.addButton(self.ui.toolButton_Foggy)
        self.weather_group.addButton(self.ui.toolButton_Dark)

        self._apply_weather_style(self.ui.toolButton_Rainy, "#6F8FAF")
        self._apply_weather_style(self.ui.toolButton_Foggy, "#9AA0A6")
        self._apply_weather_style(self.ui.toolButton_Dark, "#F5E6A1")

        self.ui.toolButton_Rainy.clicked.connect(lambda: self._select_weather("Rainy", auto_load=True))
        self.ui.toolButton_Foggy.clicked.connect(lambda: self._select_weather("Foggy", auto_load=True))
        self.ui.toolButton_Dark.clicked.connect(lambda: self._select_weather("Dark", auto_load=True))

    def _bind_buttons(self):
        self.ui.pushButton_Image.clicked.connect(self._pick_image)
        self.ui.pushButton_Video.clicked.connect(self._pick_video)
        self.ui.pushButton_Camera.clicked.connect(self._open_camera)
        self.ui.pushButton_Detect.clicked.connect(self._on_detect_clicked)
        self.ui.pushButton_Save.clicked.connect(self._on_save_clicked)

    def _on_conf_changed(self, v: int):
        self.conf = v / 100.0
        self.ui.label_Confidence.setText(f"Confidence: {self.conf:.2f}")
        if self.stream_worker:
            self.stream_worker.update_params(self.conf, self.iou)

    def _on_iou_changed(self, v: int):
        self.iou = v / 100.0
        self.ui.label_IoU.setText(f"IoU: {self.iou:.2f}")
        if self.stream_worker:
            self.stream_worker.update_params(self.conf, self.iou)

    def _select_weather(self, key: str, auto_load: bool):
        self.model_key = key
        self._set_status(f"Selected {key} model")

        if key == "Rainy":
            self.ui.toolButton_Rainy.setChecked(True)
        elif key == "Foggy":
            self.ui.toolButton_Foggy.setChecked(True)
        else:
            self.ui.toolButton_Dark.setChecked(True)

        if auto_load:
            self._load_model_for_weather(key)

    def _load_model_for_weather(self, key: str):
        path = self.MODEL_PATHS.get(key, "")
        if not path:
            QMessageBox.warning(self, "Notice", f"No model path configured for {key}.")
            return

        self.model = None
        self.model_path = path
        self.ui.label_FPS.setText("FPS: -- (loading model...)")
        self._set_status(f"{key} weather model selected. Loading...")

        self.loader = ModelLoader(path)
        self.loader.loaded.connect(self._on_model_loaded)
        self.loader.failed.connect(self._on_model_failed)
        self.loader.start()

    def _on_model_loaded(self, model, path: str):
        self.model = model
        self.model_path = path
        self.ui.label_FPS.setText("FPS: --")
        self._set_status(f"{self.model_key} weather model is ready.")

        if self.stream_worker:
            self.stream_worker.update_model(self.model, os.path.basename(path))

    def _on_model_failed(self, msg: str):
        self._set_status('Model load failed')
        QMessageBox.critical(self, "Model Load Failed", msg)
        self.ui.label_FPS.setText("FPS: --")

    def _pick_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return

        self._stop_stream_if_any()
        self.current_source_type = "image"
        self.current_image_path = path
        self.last_source_name = os.path.basename(path)
        self._set_status(f"Image selected: {self.last_source_name}")

        img = imread_unicode(path)
        if img is None:
            self._set_status('Image read failed')
            QMessageBox.warning(self, "Notice", "Failed to read image")
            return

        self._show_bgr(img)
        self._set_table_rows([])
        self.ui.label_FPS.setText("FPS: --")
        self.ui.label_Total.setText("Total: 0")

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return

        self._stop_stream_if_any()
        self.current_source_type = "video"
        self.current_video_path = path
        self.last_source_name = os.path.basename(path)
        self._set_status(f"Video selected: {self.last_source_name}")

        self.stream_worker = StreamWorker(path, is_camera=False)
        self._bind_stream_worker()
        if self.model is not None:
            self.stream_worker.update_model(self.model, os.path.basename(self.model_path))
        self.stream_worker.update_params(self.conf, self.iou)
        self.stream_worker.set_detect(False)
        self.detect_on = False
        self.ui.pushButton_Detect.setText("Detect")
        self.stream_worker.start()

    def _open_camera(self):
        self._stop_stream_if_any()
        self.current_source_type = "camera"
        self.last_source_name = "camera"
        self._set_status('Camera opened')

        self.stream_worker = StreamWorker(0, is_camera=True)
        self._bind_stream_worker()
        if self.model is not None:
            self.stream_worker.update_model(self.model, os.path.basename(self.model_path))
        self.stream_worker.update_params(self.conf, self.iou)
        self.stream_worker.set_detect(False)
        self.detect_on = False
        self.ui.pushButton_Detect.setText("Detect")
        self.stream_worker.start()

    def _bind_stream_worker(self):
        self.stream_worker.frame_out.connect(self._show_qimage)
        self.stream_worker.fps_out.connect(lambda f: self.ui.label_FPS.setText(f"FPS: {f:.1f}"))
        self.stream_worker.det_out.connect(self._set_table_rows)
        self.stream_worker.info_out.connect(self._on_stream_info)

    def _stop_stream_if_any(self):
        if self.stream_worker:
            self.stream_worker.stop()
            self.stream_worker.wait(500)
            self.stream_worker = None
        self.ui.label_FPS.setText("FPS: --")

    def _on_detect_clicked(self):
        if self.model is None:
            self._set_status('Please load model first')
            QMessageBox.warning(self, "Notice",
                                "Please select a weather model first (weights will be loaded automatically).")
            return

        if self.current_source_type == "image":
            if not self.current_image_path:
                QMessageBox.warning(self, "Notice", "Please select an image first.")
                return
            self._set_status('Detecting image...')
            self._detect_image_once(self.current_image_path)
            self._set_status('Image done')

        elif self.current_source_type in ("video", "camera"):
            if not self.stream_worker:
                QMessageBox.warning(self, "Notice", "Please open a video file or camera first.")
                return

            self.detect_on = not self.detect_on
            self.stream_worker.update_model(self.model, os.path.basename(self.model_path))
            self.stream_worker.update_params(self.conf, self.iou)
            self.stream_worker.set_detect(self.detect_on)
            self.ui.pushButton_Detect.setText("Detect (ON)" if self.detect_on else "Detect")
            self._set_status('Stream detect ON' if self.detect_on else 'Stream detect OFF')

        else:
            self._set_status('Please choose source')
            QMessageBox.warning(self, "Notice", "Please choose Image / Video / Camera first.")

    def _detect_image_once(self, path: str):
        img = imread_unicode(path)
        if img is None:
            self._set_status('Image read failed')
            QMessageBox.warning(self, "Notice", "Failed to read image")
            return

        t0 = time.perf_counter()
        res = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)[0]
        dt = time.perf_counter() - t0
        fps = 1.0 / max(dt, 1e-6)
        self.ui.label_FPS.setText(f"FPS: {fps:.1f}")

        show = res.plot()
        self._show_bgr(show)

        rows = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            names = res.names

            for i in range(len(clss)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)

                rows.append({
                    "no": i + 1,
                    "cls": names.get(int(clss[i]), str(int(clss[i]))),
                    "conf": float(confs[i]),
                    "x": float(x1),
                    "y": float(y1),
                    "w": float(w),
                    "h": float(h),
                    "model": os.path.basename(self.model_path),
                    "src": os.path.basename(path),
                })

        self._set_table_rows(rows)

    def _show_bgr(self, bgr: np.ndarray):
        self._show_qimage(bgr_to_qimage(bgr))

    def _show_qimage(self, qimg: QImage):
        self.last_qimg = qimg
        self.scene.clear()
        pix = QPixmap.fromImage(qimg)
        self.scene.addPixmap(pix)
        self.ui.graphicsView.fitInView(
            self.scene.itemsBoundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio
        )

    def _set_table_rows(self, rows: list):
        self.last_rows = rows
        self.ui.label_Total.setText(f"Total: {len(rows)}")

        t = self.ui.tableWidget
        t.setRowCount(len(rows))

        for r, it in enumerate(rows):
            item0 = QTableWidgetItem(str(it.get("cls", "")))
            item0.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            t.setItem(r, 0, item0)

            def num_item(val, fmt="{:.2f}"):
                x = QTableWidgetItem(fmt.format(val))
                x.setTextAlignment(Qt.AlignCenter)
                return x

            t.setItem(r, 1, num_item(it.get("conf", 0.0), "{:.2f}"))
            t.setItem(r, 2, num_item(it.get("x", 0.0), "{:.0f}"))
            t.setItem(r, 3, num_item(it.get("y", 0.0), "{:.0f}"))
            t.setItem(r, 4, num_item(it.get("w", 0.0), "{:.0f}"))
            t.setItem(r, 5, num_item(it.get("h", 0.0), "{:.0f}"))

    def _on_save_clicked(self):
        if self.last_qimg is None:
            self._set_status('Nothing to save')
            QMessageBox.warning(self, "Notice", "Nothing to save.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", os.getcwd())
        if not out_dir:
            self._set_status('Save cancelled')
            return

        self._set_status('Saving...')

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"cwod_result_{ts}.png"
        csv_name = f"cwod_result_{ts}.csv"

        img_path = os.path.join(out_dir, img_name)
        csv_path = os.path.join(out_dir, csv_name)

        ok = self.last_qimg.save(img_path)
        if not ok:
            QMessageBox.warning(self, "Notice", f"Failed to save image: {img_path}")
            return

        headers = ["No", "Class", "Conf", "X", "Y", "W", "H", "Model", "Source"]
        try:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(headers)
                for it in self.last_rows:
                    w.writerow([
                        it.get("no", ""),
                        it.get("cls", ""),
                        f"{it.get('conf', 0):.4f}",
                        f"{it.get('x', 0):.2f}",
                        f"{it.get('y', 0):.2f}",
                        f"{it.get('w', 0):.2f}",
                        f"{it.get('h', 0):.2f}",
                        it.get("model", ""),
                        it.get("src", self.last_source_name),
                    ])
        except Exception as e:
            QMessageBox.warning(self, "Notice", f"Failed to save CSV: {e}")
            return

        self._set_status('Saved')
        QMessageBox.information(self, "Saved", f"Saved:\n{img_path}\n{csv_path}")

    def closeEvent(self, event):
        self._stop_stream_if_any()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())
