from CellPatchExtraction.src.extraction import extract_patches
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QRadioButton, QButtonGroup
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QPixmap, QIntValidator
from PyQt5.QtCore import Qt
import PyQt5
import qimage2ndarray
import tifffile
import numpy as np
import pandas as pd
from skimage import filters
from pathlib import Path
import sys
import os

GRADES = {
    "1": "GOOD",
    "2": "MEDIUM",
    "3": "BAD",
    "4": "DIFFICULT"
}
# prevent opencv error
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(Path(PyQt5.__file__).resolve().parent/"Qt5"/"plugins")

class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("SegmentationGrader")
        self.resize(1400, 750)
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        self.input_dir = None
        self.nucleus_ending = None
        self.spots_ending = None
        self.output_dir = None
        self.in_csv_file = None
        self.out_csv_file = None
        self.segmentation_model = None
        self.segmentation_diameter = None
        self.images = None
        self.current_img_idx = None
        self.patches = None
        self.patch_length = None
        self.current_patch_idx = None
        self.df = None

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored , QSizePolicy.Policy.Ignored)
        self.image_caption_label = QLabel()
        self.image_caption_label.setAlignment(Qt.AlignCenter)
        center_layout = QVBoxLayout()
        center_layout.addStretch(50)
        center_layout.addWidget(self.image_label, 250)
        center_layout.addWidget(self.image_caption_label, 1)
        center_layout.addStretch(50)
        center_widget = QWidget()
        center_widget.setLayout(center_layout)

        self.nucleus_patch_label = QLabel()
        self.nucleus_patch_label.setAlignment(Qt.AlignCenter)
        self.nucleus_patch_label.setSizePolicy(QSizePolicy.Policy.Ignored , QSizePolicy.Policy.Ignored)
        self.spots_patch_label = QLabel()
        self.spots_patch_label.setAlignment(Qt.AlignCenter)
        self.spots_patch_label.setSizePolicy(QSizePolicy.Policy.Ignored , QSizePolicy.Policy.Ignored)
        self.combined_patch_label = QLabel()
        self.combined_patch_label.setAlignment(Qt.AlignCenter)
        self.combined_patch_label.setSizePolicy(QSizePolicy.Policy.Ignored , QSizePolicy.Policy.Ignored)
        self.patch_caption_label = QLabel()
        self.patch_caption_label.setAlignment(Qt.AlignCenter)
        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.nucleus_patch_label, 10)
        middle_layout.addWidget(self.spots_patch_label, 10)
        middle_layout.addWidget(self.combined_patch_label, 10)
        middle_layout.addWidget(self.patch_caption_label, 1)
        middle_widget = QWidget()
        middle_widget.setLayout(middle_layout)

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

        self.text_label_grade_file = QLabel("Select the file containing the grades of the images.")
        self.text_label_grade_file.setAlignment(Qt.AlignCenter)
        self.text_label_grade_file.setWordWrap(True)
        self.button_grade_file = QPushButton("Select Grade File")
        self.button_grade_file.clicked.connect(self.select_grade_file)

        self.filename_textbox = QLineEdit()
        self.filename_textbox.setPlaceholderText("Enter name of output csv-file (optional)")

        self.nucleus_ending_label = QLabel("Enter the file ending of nucleus images here...")
        self.nucleus_ending_textbox = QLineEdit("-B.TIF")
        self.spots_ending_label = QLabel("Enter the file ending of spots images here...")
        self.spots_ending_textbox = QLineEdit("-R.TIF")
        self.ending_button = QPushButton("Confirm File Endings")
        self.ending_button.clicked.connect(self.set_endings)

        self.model_selection_label = QLabel("Choose a segmentation model:")
        self.model_selection_label.setAlignment(Qt.AlignCenter)
        self.model1_button = QRadioButton("CP_BM")
        self.model2_button = QRadioButton("CP_TU")
        self.model_radio_group = QButtonGroup()
        self.model_radio_group.addButton(self.model1_button)
        self.model_radio_group.addButton(self.model2_button)
        self.model1_button.setChecked(True)
        model_button_layout = QHBoxLayout()
        model_button_layout.addStretch()
        model_button_layout.addWidget(self.model1_button)
        model_button_layout.addWidget(self.model2_button)
        model_button_layout.addStretch()
        model_button_widget = QWidget()
        model_button_widget.setLayout(model_button_layout)

        self.model_diameter_label = QLabel("Enter approximate cell diameter (0-999 pixels)")
        self.model_diameter_label.setAlignment(Qt.AlignCenter)
        self.model_diameter_input = QLineEdit("150")
        self.model_diameter_input.setValidator(QIntValidator(1, 999, self))

        self.start_button = QPushButton("Start Grading")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_grading)

        self.instruction_label = QLabel("\nRate the segmentations by pressing numbers:\n1=GOOD, 2=MEDIUM, 3=BAD, 4=DIFFICULT\nYou can undo the last grade by pressing 'e' or '0'")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        
        side_layout = QVBoxLayout()
        side_layout.addStretch()
        side_layout.addWidget(self.text_label_input)
        side_layout.addWidget(self.dir_button_input)
        side_layout.addWidget(self.text_label_output)
        side_layout.addWidget(self.dir_button_output)
        side_layout.addWidget(self.text_label_grade_file)
        side_layout.addWidget(self.button_grade_file)
        side_layout.addWidget(self.filename_textbox)
        side_layout.addWidget(self.nucleus_ending_label)
        side_layout.addWidget(self.nucleus_ending_textbox)
        side_layout.addWidget(self.spots_ending_label)
        side_layout.addWidget(self.spots_ending_textbox)
        side_layout.addWidget(self.ending_button)
        side_layout.addWidget(self.model_selection_label)
        side_layout.addWidget(model_button_widget)
        side_layout.addWidget(self.model_diameter_label)
        side_layout.addWidget(self.model_diameter_input)
        side_layout.addWidget(self.start_button)
        side_layout.addWidget(self.instruction_label)
        side_layout.addStretch()
        side_widget = QWidget()
        side_widget.setLayout(side_layout)
        side_widget.setFixedWidth(350)

        main_layout.addWidget(center_widget, 2)
        main_layout.addWidget(middle_widget, 1)
        main_layout.addWidget(side_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    

    def __exit__(self):
        self.df.to_csv(self.out_csv_file, header=not os.path.exists(self.out_csv_file), index=False)


    def resizeEvent(self, event):
        self.update_image(segment=False)
        super().resizeEvent(event)


    def keyPressEvent(self, event):
        if event.text() in {"1", "2", "3", "4"}:
            if self.images is not None and self.current_img_idx < len(self.images):
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
                
                if self.input_dir and self.output_dir and self.in_csv_file and self.nucleus_ending and self.spots_ending:
                    self.start_button.setEnabled(True)
                else:
                    self.start_button.setEnabled(False)
        return _select_directory
    

    def select_grade_file(self):
        file = QFileDialog.getOpenFileName(self, "Select Grades File")
        if file:
            self.text_label_grade_file.setText(f"Selected Grade File:\n{file[0]}")
            self.in_csv_file = file[0]
            if self.input_dir and self.output_dir and self.in_csv_file and self.nucleus_ending and self.spots_ending:
                self.start_button.setEnabled(True)
            else:
                self.start_button.setEnabled(False)
    

    def set_endings(self):
        nucleus_ending = self.nucleus_ending_textbox.text()
        spots_ending = self.spots_ending_textbox.text()
        if nucleus_ending and spots_ending:
            self.nucleus_ending = nucleus_ending.strip('.')
            self.spots_ending = spots_ending.strip('.')
            if self.input_dir and self.output_dir and self.in_csv_file and self.nucleus_ending and self.spots_ending:
                self.start_button.setEnabled(True)
            else:
                self.start_button.setEnabled(False)
        else:
            self.start_button.setEnabled(False)


    def start_grading(self):
        out_filename = self.filename_textbox.text()
        if out_filename:
            self.out_csv_file = f"{self.output_dir}/{out_filename}.csv"
        else:
            self.out_csv_file = f"{self.output_dir}/{self.input_dir.split('/')[-1]}_SEG_GRADES_{self.segmentation_model}_{self.segmentation_diameter}.csv"

        self.segmentation_model = self.model_radio_group.checkedButton().text()
        self.segmentation_diameter = int(self.model_diameter_input.text())

        print("==================================================")
        print("Starting grading...")
        print(f"Input directory: {self.input_dir}")
        print(f"Input csv: {self.in_csv_file}")
        print(f"Output csv: {self.out_csv_file}")
        print(f"Nucleus image file ending: {self.nucleus_ending}")
        print(f"Spots image file ending: {self.spots_ending}")
        print(f"Selected segmentation model: {self.segmentation_model}")
        print(f"Selected cell diameter: {self.segmentation_diameter}")
        print("==================================================")

        if os.path.exists(self.out_csv_file):
            self.df = pd.read_csv(self.out_csv_file, usecols=["img_path", "patch_num", "grade"])
        else:
            self.df = pd.DataFrame(columns=["img_path", "patch_num", "grade"])
        try:
            last_row = self.df.iloc[-1]
            last_image = last_row["img_path"]
            self.current_patch_idx = int(last_row["patch_num"]) + 1
        except IndexError:
            last_image = None
            self.current_patch_idx = 0
        self.load_images(last_image)
        self.current_img_idx = 0
        self.update_image(segment=True)


    def load_images(self, last_image):
        # get images where no channel is graded 'bad'
        grades_df = pd.read_csv(self.in_csv_file, sep=',')
        nucleus_images = grades_df.loc[(grades_df["grade"] != 3) & (grades_df["grade"] != '3') & (grades_df["img_path"].str.contains(self.nucleus_ending, na=False)), "img_path"]
        nucleus_images = set(nucleus_images.apply(lambda x: x.replace(self.nucleus_ending, "")).unique())
        spots_images = grades_df.loc[(grades_df["grade"] != 3) & (grades_df["grade"] != '3') & (grades_df["img_path"].str.contains(self.spots_ending, na=False)), "img_path"]
        spots_images = set(spots_images.apply(lambda x: x.replace(self.spots_ending, "")).unique())
        images = sorted(list(nucleus_images & spots_images))

        try:
            # resume if grading already started
            idx = images.index(last_image)
        except:
            idx = 0
        self.images = images[idx:]


    def update_image(self, segment=False):
        if self.images is None:
            return

        if self.current_img_idx >= len(self.images):
            self.image_label.setText("No images found or none left to grade!")
            self.image_caption_label.setText("")
            self.nucleus_patch_label.setText("")
            self.spots_patch_label.setText("")
            self.combined_patch_label.setText("")
            self.patch_caption_label.setText("")
        else:
            nucleus_image = tifffile.imread(self.images[self.current_img_idx] + self.nucleus_ending)
            spots_image = tifffile.imread(self.images[self.current_img_idx] + self.spots_ending)
            RGB_image = np.zeros(shape=(*nucleus_image.shape, 3), dtype=np.float32)
            RGB_image[..., 0] = spots_image
            RGB_image[..., 2] = nucleus_image

            if segment:
                self.patches = extract_patches(
                    RGB_image / np.max(RGB_image), # normalize
                    model = self.segmentation_model,
                    return_all = True,
                    patch_size = self.segmentation_diameter + 100,
                    exclude_edges = False,
                    cellpose_kwargs = {"diameter": self.segmentation_diameter}
                )
                # get number of segmented nuclei
                self.patch_length = len(self.patches["image_patches"])
                print("==================================================")
                print(f"Segmented {self.images[self.current_img_idx]}")
                print(f"Number of segmented nuclei: {self.patch_length}")
                print("==================================================")

            contours = filters.sobel(self.patches["segmentation"]==self.current_patch_idx+1)
            RGB_image[contours>0.00001] = [255, 255, 0]
            pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(RGB_image))

            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.image_caption_label.setText(self.images[self.current_img_idx].split('/')[-1])
                self.update_patch_image()


    def update_patch_image(self):
        if self.images is None or self.patches is None:
            return
        
        patch_img = self.patches["image_patches"][self.current_patch_idx]
        normalization_factor = 255 / np.max(patch_img)
        patch_img = (patch_img * normalization_factor).astype(int)

        nucleus_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(patch_img[...,2]))
        scaled_nucleus_pixmap = nucleus_pixmap.scaled(self.nucleus_patch_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.nucleus_patch_label.setPixmap(scaled_nucleus_pixmap)

        spots_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(patch_img[...,0]))
        scaled_spots_pixmap = spots_pixmap.scaled(self.spots_patch_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.spots_patch_label.setPixmap(scaled_spots_pixmap)

        combined_pixmap = QPixmap.fromImage(qimage2ndarray.array2qimage(patch_img))
        scaled_combined_pixmap = combined_pixmap.scaled(self.combined_patch_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.combined_patch_label.setPixmap(scaled_combined_pixmap)
        self.patch_caption_label.setText(f"Patch {self.current_patch_idx}")


    def grade(self, event):
        print(f"{self.images[self.current_img_idx].split('/')[-1]}   Patch {self.current_patch_idx}   {GRADES[event.text()]}")

        new_data = pd.DataFrame([{"img_path": self.images[self.current_img_idx], "patch_num": self.current_patch_idx, "grade": int(event.text())}])
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        self.df.to_csv(self.out_csv_file, index=False)

        self.current_patch_idx += 1
        if self.current_patch_idx >= self.patch_length:
            self.current_patch_idx = 0
            self.current_img_idx += 1
            self.update_image(segment=True)
        else:
            self.update_image(segment=False)
    

    def remove_grade(self):
        segment = False
        try:
            print(f"Removed grade for  {self.df.iloc[-1]['img_path'].split('/')[-1]}   Patch {self.df.iloc[-1]['patch_num']}")
            segment = self.current_patch_idx-1 < 0
            last_image = self.df.iloc[-1]['img_path']
            self.current_patch_idx = self.df.iloc[-1]['patch_num']
            self.df = self.df[:-1]
            self.df.to_csv(self.out_csv_file, index=False)
        except IndexError:
            last_image = None
            self.current_patch_idx = 0
        self.load_images(last_image)
        self.current_img_idx = 0
        self.update_image(segment=segment)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = ImageWindow()
        window.show()
        app.exec_()
    except Exception as e:
        print(f"Something went wrong! {str(e)}")
