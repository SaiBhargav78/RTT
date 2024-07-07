import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QFrame,
    QPushButton,
    QTabWidget,
    QWidget
)
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont
import pytesseract
from PIL import ImageGrab
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from Capturer import Capture, Communicator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from extractive import summarize
from googletrans import Translator
from Emotion_detection import predict_emotion
import joblib

nv_model = joblib.load("emotion_prediction_nv_model_24_april_2024.pkl")

class ScreenRegionSelector(QMainWindow):
    
    def __init__(self):
        super().__init__(None)
        # self.m_width = 300
        # self.m_height = 350

        # self.setWindowTitle("Screen Capturer")
        # self.setMinimumSize(self.m_width, self.m_height)

        # frame = QFrame()
        # frame.setContentsMargins(0, 0, 0, 0)
        # lay = QVBoxLayout(frame)
        # lay.setAlignment(Qt.AlignmentFlag.AlignJustify)
        # lay.setContentsMargins(5, 5, 5, 5)

        # self.label = QLabel()
        # self.btn_capture = QPushButton("Capture")
        # self.btn_capture.clicked.connect(self.capture)

        # lay.addWidget(self.label)
        # lay.addWidget(self.btn_capture)
        # self.setCentralWidget(frame)

        # self.capture_done = False
        self.m_width = 400
        self.m_height = 400

        self.setWindowTitle("REST APP")
        self.setMinimumSize(self.m_width, self.m_height)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.centralWidget())

        self.label = QLabel()
        self.btn_capture = QPushButton("Capture")
        self.btn_capture.clicked.connect(self.capture)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.btn_capture)

        self.capture_done = False
        self.previous_screenshot = None

    def create_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_translated_text = QWidget()
        self.tab_emotion_graph = QWidget()
        self.tab_summarized_text = QWidget()

        self.tab_widget.addTab(self.tab_translated_text, "Translated Text")
        self.tab_widget.addTab(self.tab_emotion_graph, "Emotion Graph")
        self.tab_widget.addTab(self.tab_summarized_text, "Summarized Text")

        self.setup_tab_translated_text()
        self.setup_tab_emotion_graph()
        self.setup_tab_summarized_text()

        self.layout.addWidget(self.tab_widget)

    def setup_tab_translated_text(self):
        layout = QVBoxLayout()
        self.label_translated_text = QLabel()
        layout.addWidget(self.label_translated_text)

        # Add capture button to the "Translated Text" tab
        btn_capture = QPushButton("Capture")
        btn_capture.clicked.connect(self.capture)
        layout.addWidget(btn_capture)

        self.tab_translated_text.setLayout(layout)

    def setup_tab_emotion_graph(self):
        layout = QVBoxLayout()
        # Add widgets for emotion graph display here
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Emotion')
        self.ax.set_ylabel('Probability')
        # Add capture button to the "Emotion Graph" tab
        btn_capture = QPushButton("Capture")
        btn_capture.clicked.connect(self.capture)
        layout.addWidget(btn_capture)

        self.tab_emotion_graph.setLayout(layout)

    def setup_tab_summarized_text(self):
        layout = QVBoxLayout()
        # Add widgets for summarized text display here
        self.label_summarized_text = QLabel()
        layout.addWidget(self.label_summarized_text)
        # Add capture button to the "Summarized Text" tab
        btn_capture = QPushButton("Capture")
        btn_capture.clicked.connect(self.capture)
        layout.addWidget(btn_capture)

        self.tab_summarized_text.setLayout(layout)

    
    def plot_emotion_graph(self, emotion_data):
        self.ax.clear()
        emotions = list(emotion_data.keys())
        probabilities = list(emotion_data.values())
        self.ax.bar(emotions, probabilities, align='center')
        
    def translate_text(self, text, max_chunk_length=500):
        if text == "":
            return ""

        # Split the text into smaller chunks
        # chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

        # # Translate each chunk and extract the translated text
        # translated_chunks = []
        # for chunk in chunks:
        #     translator = Translator()
        #     translation = translator.translate(chunk, dest='en')
        #     translated_text = translation.text
        #     translated_chunks.append(translated_text)
        chunks = []
        first_index = 0
        translator = Translator()
        for i in range(len(text)):
            if text[i] == "\n":
                chunk_to_translate = text[first_index:i]
                # Check if chunk_to_translate is not empty
                if chunk_to_translate.strip():
                    translation = translator.translate(chunk_to_translate, dest='en')
                    chunks.append(translation.text)
                first_index = i + 1  # Move first_index to the next character after '\n'
        # Handle the last chunk of text after the last '\n' character
        last_chunk_to_translate = text[first_index:]
        if last_chunk_to_translate.strip():
            translation = translator.translate(last_chunk_to_translate, dest='en')
            chunks.append(translation.text)
        # Combine the translated chunks into a single string
        translated_text = '\n'.join(chunks)
        return translated_text


    def capture(self):
        if not self.capture_done:
            self.create_tabs()
            self.capture_done = True
            self.capturer = Capture(self)
            self.capturer.show()
            self.capturer.communicator.release_signal.connect(self.handle_release)
            
        else:
            self.capturer = Capture(self)
            self.capturer.show()
            self.capturer.communicator.release_signal.connect(self.handle_release)

    def handle_release(self, coordinates):
        print("Mouse released at coordinates:", coordinates)
        
        x, y, w, h = coordinates
        self.previous_screenshot = ImageGrab.grab(bbox=(x,y,w,h))
        self.update_screen_text(x, y, w, h)

    # def adjust_gamma(image, gamma = 1.5):
    #     inv_gamma = 1.0/gamma
    #     table = np.array([((i/255)**inv_gamma)*255 for i in np.arange(0,256)]).astype("uint8")
    #     return cv2.LUT(image, table)
    def update_screen_text(self, x, y, w, h):
        screenshot = ImageGrab.grab(bbox=(x, y, w, h))
        if screenshot.tobytes() != self.previous_screenshot.tobytes():
            screenshot_np = np.array(screenshot)
            resized_image = cv2.resize(screenshot_np,((w-x)*2,(h-y)*2))
            # gamma_corrected = self.adjust_gamma(resized_image)
            grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(grayscale_image)
            # print(text)
            # cv2.imshow("grayscaled_image", grayscale_image)
            translated_text = self.translate_text(text)
            translated_text = translated_text.replace("/n","")
            summarized_text = summarize(translated_text)
            emotion = predict_emotion(translated_text,nv_model)
            # Clear the label content before updating it with a new capture
            self.label_translated_text.clear()
            self.label_summarized_text.clear()
            # Display the text in the label
            self.label_translated_text.setText(translated_text)
            self.label_summarized_text.setText(summarized_text)
            font = QFont()
            font.setPointSize(12)
            self.label_translated_text.setFont(font)
            self.label_summarized_text.setFont(font)
            self.plot_emotion_graph(emotion)
        
        # Restart the process by clearing the timer
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
            self.timer.deleteLater()

        # Set up a new timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.update_screen_text(x, y, w, h))
        self.timer.start(1000)


        
if __name__ == "__main__":
    app = QApplication(sys.argv)    
    app.setStyleSheet("""
    QFrame {
        background-color: #3f3f3f;
    }
                      
    QPushButton {
        border-radius: 5px;
        background-color: rgb(60, 90, 255);
        padding: 10px;
        color: white;
        font-weight: bold;
        font-family: Arial;
        font-size: 12px;
    }
                      
    QPushButton::hover {
        background-color: rgb(60, 20, 255)
    }
    """)
    selector = ScreenRegionSelector()
    selector.show()
    app.exit(app.exec_())