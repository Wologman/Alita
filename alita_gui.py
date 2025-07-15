import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import warnings
from multiprocessing import freeze_support
warnings.filterwarnings("ignore", category=UserWarning, message='A new version')

from PyQt5.QtWidgets import (QApplication,  QSizePolicy, QSystemTrayIcon,QHBoxLayout, 
                             QGraphicsDropShadowEffect, QWidget, QLabel, QLineEdit, 
                             QComboBox, QPushButton, QVBoxLayout, QFileDialog, QCheckBox,
                             QMessageBox, QMainWindow)
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtCore import Qt, QUrl, QStandardPaths
from pathlib import Path
import os
from inference import predict_images
import pandas as pd

from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QToolButton, QFrame,
    QCheckBox, QScrollArea, QGridLayout, QHBoxLayout, QLabel
)
from PyQt5.QtCore import Qt, QPoint

def get_default_dialog_dir():
    return QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or str(Path.home())

class MultiColumnDropdown(QWidget):
    def __init__(self, parent=None, items=None, columns=3):
        super().__init__(parent)
        self.setObjectName("special_animal_dropdown")

        self.items = items or []
        self.columns = columns
        self.selected = set()

        self.triggers = {
            "All Rodents": ['Mouse', 'Rat'],
            "All Ungulates": ["Deer", "Sheep", "Goat", "Pig", "Cow", "Horse", "Tahr", "Chamois"],
            "All Mustelids": ["Stoat", "Weasel", "Ferret"],
            "All Lagomorphs": ["Rabbit", "Hare"],

            "Rodentia": ['Mouse', 'Rat'],
            "Artiodactyla": ["Cervidae spp.",
                             "Ovis aries",
                             "Capra aegagrus hircus",
                             "Sus scrofa domesticus",
                             "Bos taurus",
                             "Equus ferus caballus",
                             "Hemitragus jemlahicus",
                             "Rupicapra rupicapra"],
            "Mustelidae": ["Mustela erminea", "Mustela nivalis", "Mustela furo"],
            "Lagomorpha": ["Oryctolagus cuniculus", "Lepus europaeus"],

            "Rīroi": ['Kiore Rahi', 'Kiore'],
            "Kararehe Waewae Parāoa": ["Tia", "Hipi", "Koati", "Kāpene Kuki", "Kau", "Hōiho", "(Chamois)"],
            "Matireti": ["Toriura", "Tori Uaroal", "(Ferret)"],
            "Rāpeti, Arā Rānei": ["Rāpeti", "Hea"]
        }

        self.checkbox_map = {}

        # Setup button
        self.button = QToolButton(self)
        self.button.setText("Choose Animal(s)")
        self.button.setPopupMode(QToolButton.InstantPopup)
        self.button.clicked.connect(self.toggle_popup)
        self.button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.button.setObjectName("special_animal_dropdown_button")

        # Setup popup
        self.popup = QFrame(self, Qt.Popup)
        self.popup.setFrameShape(QFrame.StyledPanel)
        self.popup.setObjectName("special_animal_dropdown_popup")

        self.popup_layout = QVBoxLayout(self.popup)
        self.popup_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)

        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(5)
        self.scroll_content.setObjectName("special_animal_dropdown_scroll")

        self.scroll_area.setWidget(self.scroll_content)
        self.popup_layout.addWidget(self.scroll_area)

        self.checkboxes = []
        self._populate_grid()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 40)
        layout.setSpacing(0)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def _populate_grid(self):
        for i, item in enumerate(self.items):
            checkbox = QCheckBox(item)
            checkbox.setObjectName("special_animal_dropdown_checkbox")
            checkbox.stateChanged.connect(self.update_selection)
            self.checkboxes.append(checkbox)
            self.checkbox_map[item] = checkbox  # NEW: Store reference
            row = i // self.columns
            col = i % self.columns
            self.grid_layout.addWidget(checkbox, row, col)

    def toggle_popup(self):
        if self.popup.isVisible():
            self.popup.hide()
        else:
            popup_width = self.button.width()
            popup_height = 600
            self.popup.resize(popup_width, popup_height)

            global_pos = self.mapToGlobal(self.button.pos())
            popup_x = global_pos.x()
            popup_y = global_pos.y() - popup_height

            self.popup.move(popup_x, popup_y)
            self.popup.show()

    def update_selection(self):
        sender = self.sender()  # NEW: Identify which checkbox triggered this
        if sender:
            text = sender.text()
            if text in self.triggers and sender.isChecked():
                for target_label in self.triggers[text]:
                    target_cb = self.checkbox_map.get(target_label)
                    if target_cb and not target_cb.isChecked():
                        target_cb.blockSignals(True)  # Avoid recursion
                        target_cb.setChecked(True)
                        target_cb.blockSignals(False)

        self.selected = {
            cb.text() for cb in self.checkboxes if cb.isChecked()
        }
        self.button.setText(", ".join(sorted(self.selected)) or "Choose Animal(s)")

    def set_items(self, new_items):
        for checkbox in self.checkboxes:
            self.grid_layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.checkboxes.clear()
        self.checkbox_map.clear()

        self.items = new_items
        self.selected.clear()
        self.button.setText("Choose Animal(s)")

        self._populate_grid()


class HoverLabel(QLabel):
    def __init__(self, text, link_text, link, parent=None):
        super().__init__(parent)
        self.link = link
        self.link_visible = link_text
        self.text_content = text  # Store the original text without HTML formatting
        self.normal_style = '<span style="color: #989898; font-weight: normal; padding:0; margin:0;">{}</span>'
        self.link_style = '<a href="{}" style="color: #989898; text-decoration: underline; padding:0; margin:0;">{}</a>'
        
        # Construct the initial HTML content
        self.setText(self.format_text(self.text_content, self.link_visible, self.link))
        self.setCursor(Qt.PointingHandCursor)  # Change cursor to hand on hover

    def format_text(self, text, link_visible, link, hover=False):
        """Helper method to format text with styles"""
        regular_text = self.normal_style.format(text)
        if hover:
            link_text = f'<a href="{link}" style="color: white; text-decoration: none;">{link_visible}</a>'
        else:
            link_text = self.link_style.format(link, link_visible)
        return f"{regular_text} {link_text}"

    def enterEvent(self, event):
        # Change link style on hover (bold and no underline)
        self.setText(self.format_text(self.text_content, self.link_visible, self.link, hover=True))
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Restore original style when mouse leaves
        self.setText(self.format_text(self.text_content, self.link_visible, self.link, hover=False))
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        QDesktopServices.openUrl(QUrl(self.link))


class HoverButton(QPushButton):
    default_style = """
                    background-color: #0d4733; 
                    color: #989898;
                    text-decoration: underline; 
                    font-weight: normal; 
                    padding:4px;
                    """
    hover_style = """
                    background-color: #0d4733; 
                    color: white; 
                    font-weight: normal; 
                    padding:4px;
                    """
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._is_shadow = True
        self.setStyleSheet(self.default_style)
        self.setCursor(Qt.PointingHandCursor)
    
    def enterEvent(self, event):
        self.setStyleSheet(self.hover_style)
        self.shadow = QGraphicsDropShadowEffect() 
        self.shadow.setBlurRadius(15) 
        self.shadow.setColor(Qt.black)
        self.setGraphicsEffect(self.shadow)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self.default_style)
        self.shadow.setEnabled(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if hasattr(self, 'click_handler'):
            self.click_handler()

    def set_click_handler(self, handler):
        self.click_handler = handler


def find_model_paths(base_dir: Path):
    models_dir = base_dir / "Models"

    # 1. Find latest *.yaml settings file in Models/Exp_*/Exp_*_Run_*.yaml
    yaml_files = sorted(
        models_dir.glob("Exp_*/Exp_*_Run_*.yaml"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    settings_pth = yaml_files[0] if yaml_files else None

    # 2. Find latest *_best_weights.pt in Models/Exp_*/Exp_*_Run_*_best_weights.pt
    weight_files = sorted(
        models_dir.glob("Exp_*/Exp_*_Run_*_best_weights.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    weights_pth = weight_files[0] if weight_files else None

    # 3. Find any .pt file directly under Models/ (excluding subdirectories)
    detector_files = sorted(
        models_dir.glob("*.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    detector_weights_pth = detector_files[0] if detector_files else None

    return {
        "settings_pth": settings_pth,
        "weights_pth": weights_pth,
        "detector_weights_pth": detector_weights_pth,
    }

class MainWindow(QMainWindow):
    def __init__(self, parent_folder):
        super().__init__()
        self.setWindowTitle("Alita | New Zealand Camera Trap Classifier")
        self.setGeometry(20, 40, 700, 950)  
        self.setObjectName("main_window")
        
        self.parent_folder = parent_folder
        self.resources = self.parent_folder / 'Resources'
        self.start_dir = get_default_dialog_dir()
        image_folder = self.resources / 'Images'
        class_names_csv = self.resources / 'name_map_edited.csv'
        df = pd.read_csv(class_names_csv, header=None)
        self.naming_schemes = list(df.iloc[0].astype(str))[1:]
        
        _class_names = {key: df.iloc[1:, idx+1].astype(str).tolist() for idx, key in enumerate(self.naming_schemes)}
        _extra_classes = {
                          'Common': ['All Rodents', 'All Mustelids', 'All Ungulates', 'All Lagomorphs'], 
                          'Scientific':['Rodentia', 'Mustelidae', 'Artiodactyla',  'Lagomorpha'],
                          'Te Reo Māori':['Rīroi', 'Matireti', 'Kararehe Waewae Parāoa', 'Rāpeti, Arā Rānei'] 
                          }
        self.class_names = {key: _extra_classes[key] + _class_names[key] for key in _class_names}
        
        self.icon_path = (image_folder / 'compuweka.ico').as_posix()
        self.grey_text = 'style="color: #989898; font-weight: Bold"'

        central_widget = QWidget()
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        logo_text = f"weka<span {self.grey_text}>Research</span>"
        self.floating_label = QLabel(logo_text, self)
        self.floating_label.setObjectName("logo")

        self.floating_label.setAlignment(Qt.AlignCenter)
        self.floating_label.setFixedSize(150, 160)
        self.floating_label.move(self.width() - self.floating_label.width() - 10, 10)
        
        self.title = QLabel("Alita")
        self.title.setObjectName("title")

        self.info_label = QLabel("Wildlife identification from camera trap images")
        self.info_label.setObjectName("info_label")

        source_1 = HoverLabel("Version 3.0 | April 2025 | ", "Documentation", "https://wekaresearch-my.sharepoint.com/:f:/p/olly/Ev_QTLzbr_pDsxJRhluRxLABH38e6ZBRO4Ig9IyBbNwG2g?e=bjrpNM")  
        source_2 = HoverLabel("|", "Source Code", "https://github.com/Wologman/Alita")
        
        self.sources_layout = QHBoxLayout()
        self.sources_layout.addWidget(source_1)
        self.sources_layout.addWidget(source_2)
        self.sources_layout.setContentsMargins(0,0,0,30)
        self.sources_layout.setSpacing(0)
        self.sources_layout.addStretch()

        self.title_layout = QVBoxLayout()
        self.title_layout.addWidget(self.title)
        self.title_layout.addWidget(self.info_label)
        self.title_layout.addLayout(self.sources_layout)
        
        self.input_label = QLabel("Folder containing the image or video files")
        self.input_path = QLineEdit(self)
        self.input_path.setPlaceholderText("Enter path here...")
        self.input_button = HoverButton("Browse for folder")
        self.input_button.set_click_handler(self.select_input_file)

        self.output_label = QLabel("Folder to save results in")
        self.output_path = QLineEdit(self)
        self.output_path.setPlaceholderText("Enter path here...")
        self.output_button = HoverButton("Browse for folder")
        self.output_button.clicked.connect(self.select_output_file)

        _thresholds_list = ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

        self.threshold_label = QLabel("Classification Threshold")
        self.threshold_help_button = QPushButton('?')
        self.threshold_help_button.setObjectName('help')
        self.threshold_help_button.clicked.connect(self.show_threshold_message)
        self.threshold_help_button.setProperty("class", "help-button")
        self.threshold_combobox =  QComboBox(self)
        self.threshold_combobox.addItems(_thresholds_list)
        self.threshold_combobox.setCurrentText('0.5')
        self.threshold_h_layout = QHBoxLayout()
        self.threshold_h_layout.addWidget(self.threshold_label)
        self.threshold_h_layout.addWidget(self.threshold_help_button)
        self.threshold_h_layout.setContentsMargins(0,0,0,0)
        self.threshold_h_layout.addStretch()
        self.threshold_v_layout = QVBoxLayout()
        self.threshold_v_layout.addLayout(self.threshold_h_layout)
        self.threshold_v_layout.addWidget(self.threshold_combobox)


        _core_list = [1, 2, 4, 6, 8, 12, 16, 24, 36]
        _available_cores = os.cpu_count() or 1
        _core_list = [str(x) for x in _core_list if x <= _available_cores // 2]
        _default_cores = str(_core_list[1] if len(_core_list) >= 2 else _core_list[0])

        self.cores_label = QLabel("Number of CPU cores to use")
        self.cores_help_button = QPushButton('?')
        self.cores_help_button.setObjectName('cores_help')
        self.cores_help_button.clicked.connect(self.show_cores_message)
        self.cores_help_button.setProperty("class", "help-button")
        self.cores_combobox =  QComboBox(self)
        self.cores_combobox.addItems(_core_list)
        self.cores_combobox.setCurrentText(_default_cores)
        self.cores_h_layout = QHBoxLayout()
        self.cores_h_layout.addWidget(self.cores_label)
        self.cores_h_layout.addWidget(self.cores_help_button)
        self.cores_h_layout.setContentsMargins(0,0,0,0)
        self.cores_h_layout.addStretch()
        self.cores_v_layout = QVBoxLayout()
        self.cores_v_layout.addLayout(self.cores_h_layout)
        self.cores_v_layout.addWidget(self.cores_combobox)

        self.gpu_checkbox = QCheckBox("Run on GPU", self)
        self.gpu_help_button = QPushButton('?')
        self.gpu_help_button.setObjectName('gpu_help')
        self.gpu_help_button.clicked.connect(self.show_gpu_message)
        self.gpu_help_button.setProperty("class", "help-button")
        self.gpu_h_layout = QHBoxLayout()
        self.gpu_h_layout.addWidget(self.gpu_checkbox)
        self.gpu_h_layout.addWidget(self.gpu_help_button)
        self.gpu_h_layout.setContentsMargins(0,0,0,0)
        self.gpu_h_layout.addStretch()
        self.gpu_v_layout = QVBoxLayout()
        self.gpu_v_layout.addLayout(self.gpu_h_layout)

        self.naming_label = QLabel("Naming Scheme")
        self.naming_combo = QComboBox(self)
        self.naming_combo.addItems(self.naming_schemes)
        self.naming_combo.currentIndexChanged.connect(self.naming_changed)
        self.naming_combo.setCurrentIndex(0)  # Use this instead of setCurrentText
        #self.naming_combo.setCurrentText(self.naming_schemes[0])
        

        self.special_animal_label = QLabel("Animals of special interest")
        self.special_animal_help_button = QPushButton('?')
        self.special_animal_help_button.setObjectName('special_animal_help')
        self.special_animal_help_button.setProperty("class", "help-button")
        self.special_animal_help_button.clicked.connect(self.show_special_animal_message)

        #self.special_animal_combo = QComboBox(self)
        #self.special_animal_combo.addItems(["All"] + self.birdnames.short_names_list)
        self.special_animal_dropdown = MultiColumnDropdown(items=self.class_names[self.naming_schemes[0]], columns=3)
        self.special_animal_h_layout = QHBoxLayout()
        self.special_animal_h_layout.addWidget(self.special_animal_label)
        self.special_animal_h_layout.addWidget(self.special_animal_help_button)
        self.special_animal_h_layout.setContentsMargins(0,0,0,0)
        self.special_animal_h_layout.addStretch()
        self.run_button = HoverButton("Run Alita")
        self.run_button.setObjectName('run')
        self.run_button.clicked.connect(self.run_program)

        attribution_text = "Made in Nelson for the Department of Conservation - Te Papa Atawhai | By " 
        self.attribution = HoverLabel(attribution_text, "wekaResearch", "https://wekaresearch.com")

        layout.addLayout(self.title_layout)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_path)
        layout.addWidget(self.input_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_path)
        layout.addWidget(self.output_button)
        layout.addLayout(self.threshold_v_layout)
        layout.addLayout(self.cores_v_layout)
        layout.addLayout(self.gpu_v_layout)
        layout.addWidget(self.naming_label)
        layout.addWidget(self.naming_combo)
        layout.addLayout(self.special_animal_h_layout)
        layout.addWidget(self.special_animal_dropdown)
        layout.addWidget(self.run_button)
        layout.addStretch() 
        layout.addWidget(self.attribution)


    def show_threshold_message(self):
        QMessageBox.information(self, "Classification Threshold", 
                                "This effects the 'empty' and 'unknown' predictions. "
                                "A threshold of 0.5 is reasonable, for most locations \n\n"

                                "If no class makes the threshold, and the Detection stage already predicted "
                                "there should be an animal, then the prediction will be 'Unknown'.\n\n"

                                "If both Megadetector, and the max score from the classifier are below their "
                                "respective thresholds then the image will be classed as 'Empty'\n\n"
                                
                                "Setting a higher threshold may improve precision, but at the expense of recall. "
                                "This could be more appropriate for long term population monitoring.\n\n"
                                
                                "Setting a lower threshold may improve recall, but at the expense of precision. "
                                "This might be appropriate for predator incursion "
                                "surveilance in conjunction with manual inspection for false positives.\n\n"
                                )

    def show_cores_message(self):
        QMessageBox.information(self, "Number of cores", 
                                "The number of CPU cores to be used for parallel processing.\n\n"
                                "By default the menu allows up to half your thread count.\n"
                                "Generally the more cores the faster the processing. But\n"
                                "more cores also increases the chance of a crash due to insufficient memory.\n\n"
                                "Be careful if you want to use your machine for other stuff at the same time.\n\n"
                                "Set to the max CPU limit if you're feeling lucky.... punk."
                                )

    def show_gpu_message(self):
        QMessageBox.information(self, "GPU usage", 
                                "If you tick this box, the program will try to use your GPU for increased speed.\n\n"
                                "Only an NVIDIA GPU will work, and the program will crash if it has insufficient memory. "
                                "If you do get an out-of-memory error, try backing off the CPU count.\n\n"
                                "The program was tested successfully on a Dell 3420 using an an NVIDIA MX450 GPU, whilst "
                                "set to use 2 CPU cores."
                                )

    def show_special_animal_message(self):
        QMessageBox.information(self, "Animals of Interst",
            "You usually wouldn't need this.  It is for the situation where you " \
            "are only interested in monitoring a single species.\n\n"
            "Each animal class selected will produce two additional files:\n\n"
            "1. A .json file for visualisation in Timelapse (https://timelapse.ucalgary.ca) "
            "The score for the animal of interest will be given for "
            "every image, for filtering in Timelapse.\n\n"
            "2. A .csv file with a column listing all the file paths "
            "where predictions were over the threshold, plus a second column "
            "with the prediction score."
            )

    def resizeEvent(self, event):
        self.floating_label.move(self.width() - self.floating_label.width() - 10, 10)
        
    def select_input_file(self):
        file_dialog = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.start_dir)
        if os.path.isdir(file_dialog):
            self.input_path.setText(file_dialog)

    def select_output_file(self):
        file_dialog = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.start_dir)
        if os.path.isdir(file_dialog):
            self.output_path.setText(file_dialog)

    def naming_changed(self):
        selected_index = self.naming_combo.currentIndex()
        self.special_animal_dropdown.set_items(self.class_names[self.naming_schemes[selected_index]])
        #self.special_animal_dropdown.addItems(["All", "rats", 'mice']) # + self.class_names[self.naming_schemes[selected_index]]))




    def run_program(self):
        if __name__ == "__main__":
            input_folder = self.input_path.text()
            output_folder = self.output_path.text()
            threshold = float(self.threshold_combobox.currentText())
            num_cores = self.cores_combobox.currentText()
            use_cpu = not bool(self.gpu_checkbox.isChecked())
            naming_index = self.naming_combo.currentIndex()
            chosen_classes = self.special_animal_dropdown.selected  #Something wrong with this

            if not input_folder or not output_folder:
                msg = QMessageBox.critical(self, "Please specify both input and output folder paths.")
                return
            if not os.path.isdir(input_folder):
                msg = QMessageBox.critical(self, "The input folder is not a valid directory, please chose another one.")
                return
            if not os.path.isdir(output_folder):
                msg = QMessageBox.critical(self, "Nearly there", "The output folder path is not a valid directory, please chose another location.")
                return
            
            model_paths = find_model_paths(self.parent_folder)

            arguments = {
                    'project_dir': self.parent_folder,
                    'image_dir': input_folder,
                    'settings_pth': model_paths['settings_pth'],
                    'weights_pth': model_paths['weights_pth'],
                    'detector_weights_pth': model_paths['detector_weights_pth'],
                    'md_empty_threshold': 0.15,
                    'classify_conf_threshold': threshold,
                    'predictions_dir': output_folder,
                    'naming_scheme': self.naming_schemes[naming_index],
                    'cpu_only': use_cpu,
                    'num_workers': int(num_cores),
                    'special_interest_classes': chosen_classes,
                    }

            try:
                df, speed = predict_images(**arguments)
                num_preds = len(df)
                if num_preds > 0: 
                    QMessageBox.information(self, f"Yay, Success", f"Alita completed processing {num_preds} files!")
                if num_preds == 0:
                    QMessageBox.information(self, "Hmmmm", "Alita completed processing but no predictions were made.  Was the folder empty?")
            except Exception as e:
                QMessageBox.critical(self, "Bummer", f"For some mysterious reason the program failed. The error code was: {e}")


import tempfile
def get_project_root() -> Path:
    """
    Determine project root depending on execution context:
    - --onefile: sys._MEIPASS will be a temp directory
    - --onedir: use parent of executable (then go up one)
    - dev mode: use __file__
    """

    if getattr(sys, 'frozen', False):
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass and Path(meipass).resolve().parent == Path(tempfile.gettempdir()).resolve():
            # This is really --onefile mode
            return Path(meipass).resolve()  # type: ignore[attr-defined]
        else:
            # This is really --onedir mode
            return Path(sys.executable).resolve().parent.parent
    else:
        return Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    freeze_support()  #I would like to understand why I don't need this for Kaytoo

    root = get_project_root()
    print(f"Project root is: {root}")
    css_path = root / 'Resources/alita_gui_styles.css'
    QDir.addSearchPath('images', str(root / 'Resources/Images'))

    app = QApplication(sys.argv)
    with open(css_path, 'r') as file:
        app.setStyleSheet("")
        app.setStyleSheet(file.read())
    window = MainWindow(root)
    tray_icon = QSystemTrayIcon()
    icon = QIcon(window.icon_path)
    tray_icon.setIcon(icon)
    tray_icon.show()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec_())