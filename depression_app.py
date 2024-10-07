import sys
import pyaudio
import wave
import speech_recognition as sr
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, 
                             QTextEdit, QFrame, QHBoxLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QPalette, QColor
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

class DepressionPredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.recognizer = sr.Recognizer()
        
        # Load the model and tokenizer
        self.model = load_model('depression_model.h5')
        print("Model input shape:", self.model.input_shape)
        with open('tokenizer.pkl', 'rb') as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)
        
        # Set the max sequence length based on the model's input shape
        self.max_sequence_length = self.model.input_shape[1]
        print("Max sequence length:", self.max_sequence_length)

    def initUI(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: none;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QTextEdit {
                background-color: #3a3a3a;
                border: 1px solid #5a5a5a;
                border-radius: 5px;
            }
            QLabel {
                padding: 5px;
            }
        """)

        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel('Depression Prediction App')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        main_layout.addWidget(title_label)

        # Recording controls
        control_layout = QHBoxLayout()
        self.start_button = QPushButton('Start Recording')
        self.start_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.start_button)

        main_layout.addLayout(control_layout)

        # Transcription
        transcription_frame = QFrame()
        transcription_frame.setStyleSheet("background-color: #3a3a3a; border-radius: 10px; padding: 10px;")
        transcription_layout = QVBoxLayout(transcription_frame)

        self.transcription_label = QLabel('Transcription:')
        transcription_layout.addWidget(self.transcription_label)

        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        transcription_layout.addWidget(self.transcription_text)

        main_layout.addWidget(transcription_frame)

        # Prediction
        self.prediction_label = QLabel('Depression Prediction: N/A')
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-top: 20px;")
        main_layout.addWidget(self.prediction_label)

        self.setLayout(main_layout)
        self.setWindowTitle('Depression Prediction App')
        self.setGeometry(300, 300, 500, 400)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.start_button.setText('Stop Recording')
        self.start_button.setStyleSheet("background-color: #8b0000;")  # Dark red color
        self.frames = []

        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        self.timer = QTimer()
        self.timer.timeout.connect(self.record_frame)
        self.timer.start(10)  # Record every 10ms

    def record_frame(self):
        data = self.stream.read(1024)
        self.frames.append(data)

    def stop_recording(self):
        self.is_recording = False
        self.start_button.setText('Start Recording')
        self.start_button.setStyleSheet("")  # Reset to default style
        self.timer.stop()
        self.stream.stop_stream()
        self.stream.close()

        # Save the recorded audio
        wf = wave.open('recorded_audio.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Transcribe the audio
        transcription = self.transcribe_audio('recorded_audio.wav')
        self.transcription_text.setText(transcription)

        # Predict depression
        prediction = self.predict_depression(transcription)
        self.prediction_label.setText(f'Depression Prediction: {prediction:.2f}%')

    def transcribe_audio(self, audio_file):
        with sr.AudioFile(audio_file) as source:
            audio = self.recognizer.record(source)
        try:
            return self.recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Speech recognition could not understand the audio"
        except sr.RequestError:
            return "Could not request results from the speech recognition service"

    def predict_depression(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length)
        
        raw_prediction = self.model.predict(padded_sequence)[0][0]
        print(f"Raw prediction: {raw_prediction}")
        
        prediction = raw_prediction * 100  # Convert to percentage
        return prediction

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DepressionPredictionApp()
    ex.show()
    sys.exit(app.exec_())