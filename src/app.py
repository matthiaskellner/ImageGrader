import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from glob import glob
import qimage2ndarray
import tifffile
import pandas as pd
import os

GRADES = {
    "1": "GOOD",
    "2": "MEDIUM",
    "3": "BAD"
}

class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ImageGrader")
        self.resize(1200, 700)
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        self.input_dir = None
        self.ending = None
        self.output_dir = None
        self.csv_file = None
        self.images = None
        self.current_idx = None
        self.df = None

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored , QSizePolicy.Policy.Ignored)
        self.image_caption_label = QLabel()
        self.image_caption_label.setAlignment(Qt.AlignCenter)
        
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.image_label, 10)
        center_layout.addWidget(self.image_caption_label, 0)
        center_layout.addStretch(1)
        center_widget = QWidget()
        center_widget.setLayout(center_layout)

        self.text_label_input = QLabel("Select the input directory using the button below.")
        self.text_label_input.setAlignment(Qt.AlignCenter)
        self.text_label_input.setWordWrap(True)
        self.dir_button_input = QPushButton("Select Input Directory")
        self.dir_button_input.clicked.connect(self.select_directory("input"))

        self.text_label_output = QLabel("Select the output directory using the button below.")
        self.text_label_output.setAlignment(Qt.AlignCenter)
        self.text_label_output.setWordWrap(True)
        self.dir_button_output = QPushButton("Select Output Directory")
        self.dir_button_output.clicked.connect(self.select_directory("output"))

        self.filename_textbox = QLineEdit()
        self.filename_textbox.setPlaceholderText("Enter name of output csv-file (optional)")

        self.ending_textbox = QLineEdit()
        self.ending_textbox.setPlaceholderText("Enter the file ending of images here...")
        self.ending_button = QPushButton("Confirm File Ending")
        self.ending_button.clicked.connect(self.set_ending)

        self.start_button = QPushButton("Start Grading")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_grading)

        self.instruction_label = QLabel("\nRate the images by pressing numbers:\n1=GOOD, 2=MEDIUM, 3=BAD\n\nYou can undo the last grade by pressing 'e' or '0'")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        
        side_layout = QVBoxLayout()
        side_layout.addStretch()
        side_layout.addWidget(self.text_label_input)
        side_layout.addWidget(self.dir_button_input)
        side_layout.addWidget(self.text_label_output)
        side_layout.addWidget(self.dir_button_output)
        side_layout.addWidget(self.filename_textbox)
        side_layout.addWidget(self.ending_textbox)
        side_layout.addWidget(self.ending_button)
        side_layout.addWidget(self.start_button)
        side_layout.addWidget(self.instruction_label)
        side_layout.addStretch()
        side_widget = QWidget()
        side_widget.setLayout(side_layout)
        side_widget.setFixedWidth(350)

        main_layout.addWidget(center_widget)
        main_layout.addWidget(side_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    

    def __exit__(self):
        self.df.to_csv(self.csv_file, header=not os.path.exists(self.csv_file), index=False)
    

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)


    def keyPressEvent(self, event):
        if event.text() in {"1", "2", "3"}:
            if self.images is not None and self.current_idx < len(self.images):
                self.grade(event)
        elif event.text() in {"0", "e"}:
            self.remove_grade()


    def select_directory(self, type):
        def _select_directory():
            dir = QFileDialog.getExistingDirectory(self, "Select Directory")
            if dir:
                if type == "input":
                    self.text_label_input.setText(f"Selected Input Directory:\n{dir}")
                    self.input_dir = dir
                else:
                    self.text_label_output.setText(f"Selected Output Directory:\n{dir}")
                    self.output_dir = dir
                
                if self.input_dir and self.output_dir and self.ending:
                    self.start_button.setEnabled(True)
                else:
                    self.start_button.setEnabled(False)
        return _select_directory
    
    def set_ending(self):
        ending = self.ending_textbox.text()
        if ending:
            self.ending = ending.strip('.')
            if self.input_dir and self.output_dir and self.ending:
                self.start_button.setEnabled(True)
            else:
                self.start_button.setEnabled(False)
        else:
            self.start_button.setEnabled(False)


    def start_grading(self):
        filename = self.filename_textbox.text()
        if filename:
            self.csv_file = f"{self.output_dir}/{filename}.csv"
        else:
            self.csv_file = f"{self.output_dir}/{self.input_dir.split('/')[-1]}_GRADES.csv"

        print("==================================")
        print("Starting grading...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output csv: {self.csv_file}")
        print(f"Image file ending: {self.ending}")
        print("==================================")

        if os.path.exists(self.csv_file):
            self.df = pd.read_csv(self.csv_file, usecols=["img_path", "grade"])
        else:
            self.df = pd.DataFrame(columns=["img_path", "grade"])
        try:
            last_image = self.df.iloc[-1]["img_path"]
        except IndexError:
            last_image = None
        self.load_images(last_image)
        self.current_idx = 0
        self.update_image()


    def load_images(self, last_img):
        images = sorted(glob(f"{self.input_dir}/*.{self.ending}"))
        try:
            # resume if grading already started
            idx = images.index(last_img) + 1
        except:
            idx = 0
        self.images = images[idx:]


    def update_image(self):
        if self.images is None:
            return
        
        if self.current_idx >= len(self.images):
            self.image_label.setText("No images found or none left to grade!")
            self.image_caption_label.setText("")
        else:
            image = tifffile.imread(self.images[self.current_idx])
            pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(image))

            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.image_caption_label.setText(self.images[self.current_idx].split('/')[-1])

    
    def grade(self, event):
        print(f"{self.images[self.current_idx].split('/')[-1]}    {GRADES[event.text()]}")
        new_data = pd.DataFrame([{"img_path": self.images[self.current_idx], "grade": int(event.text())}])
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        self.df.to_csv(self.csv_file, index=False)
        self.current_idx += 1
        self.update_image()
    
    def remove_grade(self):
        try:
            print(f"Removed grade for  {self.df.iloc[-1]['img_path'].split('/')[-1]}")
            self.df = self.df[:-1]
            self.df.to_csv(self.csv_file, index=False)
            last_image = self.df.iloc[-1]["img_path"]
        except IndexError:
            last_image = None
        self.load_images(last_image)
        self.current_idx = 0
        self.update_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = ImageWindow()
        window.show()
        app.exec_()
    except Exception as e:
        print(f"Something went wrong! {str(e)}")
