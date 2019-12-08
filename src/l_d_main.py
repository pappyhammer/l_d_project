from qtpy.QtWidgets import *
from qtpy import QtGui
from qtpy import QtCore
from qtpy.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import os
import PIL
from ScanImageTiffReader import ScanImageTiffReader
from PIL import ImageSequence
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import sys
import platform
import os


# class ZStackImages

def get_tiff_names(red_dir_path, cfos_dir_path, mask_dir_path, verbose=False):
    """
    Return a dict with: group, f, position, s, depth

    :param red_dir_path:
    :param cfos_dir_path:
    :param mask_dir_path:
    :return:
    """
    results_dict = dict()

    to_explore = {"red": red_dir_path, "cfos": cfos_dir_path, "mask": mask_dir_path}

    for tiff_key, dir_path in to_explore.items():
        file_names = []
        # look for filenames in the fisrst directory, if we don't break, it will go through all directories
        for (dirpath, dirnames, local_filenames) in os.walk(dir_path):
            file_names.extend(local_filenames)
            break

        # now we parse all file_names and distribute them in the right category
        # typical file_name red-GroupA-F1-dors-s1-dist.tif
        for file_name in file_names:
            index_group = file_name.index("-")
            file_name_cropped = file_name[index_group+1:]
            index_f = file_name_cropped.index("-")
            group = file_name_cropped[:index_f]
            if group not in results_dict:
                results_dict[group] = dict()
            if verbose:
                print(f"group {group}")
            file_name_cropped = file_name_cropped[index_f + 1:]
            index_pos = file_name_cropped.index("-")
            f_value = file_name_cropped[:index_pos]
            if f_value not in results_dict[group]:
                results_dict[group][f_value] = dict()
            if verbose:
                print(f"f_value {f_value}")
            file_name_cropped = file_name_cropped[index_pos + 1:]
            index_s = file_name_cropped.index("-")
            pos_value = file_name_cropped[:index_s]
            if pos_value not in results_dict[group][f_value]:
                results_dict[group][f_value][pos_value] = dict()
            if verbose:
                print(f"pos_value {pos_value}")
            file_name_cropped = file_name_cropped[index_s + 1:]
            index_depth = file_name_cropped.index("-")
            s_value = file_name_cropped[:index_depth]
            if s_value not in results_dict[group][f_value][pos_value]:
                results_dict[group][f_value][pos_value][s_value] = dict()
            if verbose:
                print(f"s_value {s_value}")
            file_name_cropped = file_name_cropped[index_depth + 1:]
            try:
                index_end = file_name_cropped.index("_")
            except ValueError:
                index_end = file_name_cropped.index(".")
            depth_value = file_name_cropped[:index_end]
            if depth_value not in results_dict[group][f_value][pos_value][s_value]:
                results_dict[group][f_value][pos_value][s_value][depth_value] = dict()
            if verbose:
                print(f"depth_value {depth_value}")
                print("")
            results_dict[group][f_value][pos_value][s_value][depth_value][tiff_key] = os.path.join(dir_path, file_name)
    return results_dict


class MainWindow(QMainWindow):
    """Main window of the Exploratory GUI"""
    def __init__(self):
        super().__init__(parent=None)

        self.setWindowTitle("cFos GUI")

        screenGeometry = QApplication.desktop().screenGeometry()
        # making sure the window is not bigger than the dimension of the screen
        width_window = min(1800, screenGeometry.width())
        # width_window = screenGeometry.width()
        height_window = min(1000, screenGeometry.height())
        self.resize(width_window, height_window)

        ## creating widgets to put in the window
        self.central_widget = CentralWidget(main_window=self)
        self.setCentralWidget(self.central_widget)

        self.show()

class CentralWidget(QWidget):

    def __init__(self, main_window):
        super().__init__(parent=main_window)

        root_path = "/Users/pappyhammer/Documents/academique/these_inmed/Lexi_Davide_project/"
        result_path = os.path.join(root_path, "results_ld")

        mask_dir_path = os.path.join(root_path, "masques")
        red_dir_path = os.path.join(root_path, "cellules (red)")
        cfos_dir_path = os.path.join(root_path, "cfos (green)")

        tiffs_dict = get_tiff_names(red_dir_path=red_dir_path, cfos_dir_path=cfos_dir_path, mask_dir_path=mask_dir_path)
        raise Exception("KING IN THE NORTH")

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        current_z = 0

        self.main_layout = QHBoxLayout()

        self.grid_layout = QGridLayout()

        cells_widget = CellsDisplayMainWidget(current_z=current_z)
        self.grid_layout.addWidget(cells_widget, 0, 0)

        cfos_widget = CellsDisplayMainWidget(current_z=current_z)
        self.grid_layout.addWidget(cfos_widget, 0, 1)

        mask_widget = CellsDisplayMainWidget(current_z=current_z)
        self.grid_layout.addWidget(mask_widget, 1, 0)

        overlap_widget = CellsDisplayMainWidget(current_z=current_z)
        self.grid_layout.addWidget(overlap_widget, 1, 1)

        self.main_layout.addLayout(self.grid_layout)

        self.setLayout(self.main_layout)



class CellsDisplayMainWidget(pg.GraphicsLayoutWidget):
    """
    Module that will display the different w intervals along the frames
    """

    def __init__(self, current_z, parent=None):

        # self.view_box = MyViewBox()
        pg.GraphicsLayoutWidget.__init__(self) # viewBox=self.view_box
        # allows the widget to be expanded in both axis
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.current_z = current_z

        view = self.addViewBox(lockAspect=True, row=0, col=0, invertY=True)
        view.setMenuEnabled(False)

        self.image_displayed = pg.ImageItem(axisOrder='row-major', border='w')
        view.addItem(self.image_displayed)

        # self.image_displayed.setImage(data)

    def keyPressEvent(self, event):
        """
        Call when a key is pressed
        Args:
            event:

        Returns:

        """
        # Sending the event to the main window if the widget is in the main window
        if self.main_window is not None:
            self.main_window.keyPressEvent(event=event)

class PlanMask:

    def __init__(self, mask_data, mask_id, result_path):
        self.mask_data = mask_data
        self.mask_id = mask_id
        self.result_path = result_path
        # plt.imshow(mask_data)
        # plt.show()
        mask_with_contours = mask_data.copy()
        mask_with_contours = np.reshape(mask_with_contours, (mask_with_contours.shape[0],mask_with_contours.shape[1], 1 ))
        # contours, hierarchy = cv2.findContours(mask_with_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #
        contours, hierarchy = cv2.findContours(mask_with_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.all_contours = contours
        self.n_cells = len(self.all_contours)
        # print(f"contours {contours}")
        # Draw all contours
        # -1 signifies drawing all contours
        # cv2.drawContours(mask_with_contours, contours, -1, (255, 0, 0), 1)

        # cv2.imshow('Contours', mask_with_contours)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def plot_with_contours(self, save_formats="png"):
        background_color = "black"

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(20, 20), dpi=200)
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        plt.tight_layout()

        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        ax.imshow(self.mask_data, cmap=plt.get_cmap("Reds"))

        for contours in self.all_contours:
            # print(f"contours {contours}")
            # print(f"contours shape {contours.shape}")
            xy = []
            for c in contours:
                xy.append([c[0][0], c[0][1]])
            # xy = self.coords[cell].transpose()
            cell_polygon = patches.Polygon(xy=xy,
                                                fill=False, linewidth=1,
                                                facecolor=None,
                                                edgecolor="yellow",
                                                zorder=10)  # lw=2
            ax.add_patch(cell_polygon)

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{self.result_path}/{self.mask_id}_{self.n_cells}_cells.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())

        plt.close()


def main():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/Lexi_Davide_project/"
    result_path = os.path.join(root_path, "results_ld")

    masks_path = os.path.join(root_path, "masques")

    mask_id = "red-GroupA-F1-dors-s1-dist"
    mask_data = load_mask(masks_path=masks_path, mask_id=mask_id)

    for index in range(len(mask_data)):
        plan_mask = PlanMask(mask_data=mask_data[index], result_path=result_path, mask_id=mask_id + f"_{index}")
        plan_mask.plot_with_contours()



def load_mask(masks_path, mask_id):
    """
    Return the 3 mask dist - - in a 3d np array (1: dist-..., tiff data)
    :param masks_path:
    :param mask_id:
    :return:
    """
    "mid, prox"

    mask_data = None

    # layers = ["dist", "mid", "prox"]

    file_name = os.path.join(masks_path, mask_id + ".tif")
    try:
        layer_data = ScanImageTiffReader(file_name).data()

    except Exception as e:
        im = PIL.Image.open(file_name)
        print(f"np.array(im).shape {np.array(im).shape}, np.max {np.max(np.array(im))}")
        layer_data = ImageSequence.Iterator(im)[0]
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_y, dim_x = np.array(im).shape
    print(f"layer_data.shape {layer_data.shape}, np.max {np.max(layer_data)}")
    # if mask_data is None:
    #     #     mask_data = np.zeros((3 * 7, layer_data.shape[1], layer_data.shape[2]), dtype="uint8")
    #     # for index in np.arange(layer_data.shape[0]):
    #     #     mask_data[layer_index * layer_data.shape[0] + index] = layer_data[index]
    return layer_data

def main_gui():
    app = QApplication(sys.argv)

    # set the environment variable to use a specific wrapper
    # it can be set to PyQt, PyQt5, PySide or PySide2 (not implemented yet)
    # os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

    # dark_style_style_sheet = qdarkstyle.load_stylesheet_from_environment(is_pyqtgraph=True)
    # from package qdarkstyle, modified css
    my_path = os.path.abspath(os.path.dirname(__file__))
    if platform.system() == "Windows":
        to_insert = os.path.join(my_path, "icons/")
        to_insert = to_insert.replace("\\", "/")
    else:
        to_insert = os.path.join(my_path, "icons/")

    file_name = os.path.join(my_path, "cicada_qdarkstyle.css")
    # with open(file_name, "w", encoding='UTF-8') as file:
    #     file.write(dark_style_style_sheet)
    with open(file_name, "r", encoding='UTF-8') as file:
        dark_style_cicada_style_sheet = file.read()

    dark_style_cicada_style_sheet = dark_style_cicada_style_sheet.replace("icons/", to_insert)
    app.setStyleSheet(dark_style_cicada_style_sheet)

    # config_handler = ConfigHandler()

    main_window = MainWindow()

    # putting the window at the center of the screen
    # screenGeometry is an instance of Qrect
    screenGeometry = QApplication.desktop().screenGeometry()
    x = (screenGeometry.width() - main_window.width()) / 2
    y = (screenGeometry.height() - main_window.height()) / 2
    main_window.move(x, y)
    main_window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main_gui()
    # main()
