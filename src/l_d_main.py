from qtpy.QtWidgets import *
from qtpy import QtGui
from qtpy import QtCore
from qtpy.QtCore import Qt
import pyqtgraph as pg
from PyQt5 import QtCore as Core
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
from sortedcontainers import SortedDict
from l_d_rois import PolyLineROI


# class ZStackImages

def get_image_from_tiff(file_name):
    """

    Args:
        file_name:

    Returns:

    """
    try:
        layer_data = ScanImageTiffReader(file_name).data()

    except Exception as e:
        im = PIL.Image.open(file_name)
        print(f"np.array(im).shape {np.array(im).shape}, np.max {np.max(np.array(im))}")
        layer_data = ImageSequence.Iterator(im)[0]
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_y, dim_x = np.array(im).shape


    # print(f"layer_data.shape {layer_data.shape}, np.max {np.max(layer_data)}, np.min{np.max(layer_data)}")
    return layer_data


def get_tiff_names(red_dir_path, cfos_dir_path, mask_dir_path, verbose=False):
    """
    Return a dict with: group, f, position, s, depth

    :param red_dir_path:
    :param cfos_dir_path:
    :param mask_dir_path:
    :return:
    """
    results_dict = SortedDict()

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
                results_dict[group] = SortedDict()
            if verbose:
                print(f"group {group}")
            file_name_cropped = file_name_cropped[index_f + 1:]
            index_pos = file_name_cropped.index("-")
            f_value = file_name_cropped[:index_pos]
            if f_value not in results_dict[group]:
                results_dict[group][f_value] = SortedDict()
            if verbose:
                print(f"f_value {f_value}")
            file_name_cropped = file_name_cropped[index_pos + 1:]
            index_s = file_name_cropped.index("-")
            pos_value = file_name_cropped[:index_s]
            if pos_value not in results_dict[group][f_value]:
                results_dict[group][f_value][pos_value] = SortedDict()
            if verbose:
                print(f"pos_value {pos_value}")
            file_name_cropped = file_name_cropped[index_s + 1:]
            index_depth = file_name_cropped.index("-")
            s_value = file_name_cropped[:index_depth]
            if s_value not in results_dict[group][f_value][pos_value]:
                results_dict[group][f_value][pos_value][s_value] = SortedDict()
            if verbose:
                print(f"s_value {s_value}")
            file_name_cropped = file_name_cropped[index_depth + 1:]
            try:
                index_end = file_name_cropped.index("_")
            except ValueError:
                index_end = file_name_cropped.index(".")
            depth_value = file_name_cropped[:index_end]
            if depth_value not in results_dict[group][f_value][pos_value][s_value]:
                results_dict[group][f_value][pos_value][s_value][depth_value] = SortedDict()
            if verbose:
                print(f"depth_value {depth_value}")
                print("")
            results_dict[group][f_value][pos_value][s_value][depth_value][tiff_key] = os.path.join(dir_path, file_name)
    return results_dict


class RoisManager:

    def __init__(self, rois_manager_id, n_displays, cells_display_keys, cells_display_dict):

        # each roi has a cell_id
        self.rois_by_layer_dict = dict()
        # how many displays, one will have modifiable ROIs, the others will be link to the modifiable one
        self.n_displays = n_displays
        # rois_id: tuple of strings
        self.rois_manager_id = rois_manager_id
        # first one is for the mask
        self.cells_display_keys = cells_display_keys
        # contains the instance of CellsDisplayMainWidget
        self.cells_display_dict = cells_display_dict
        # there because of PolyLineRoi instances
        self.display_rois = True
        # to keep them unique
        self.individual_roi_id = 0

    def get_pg_rois(self, cells_display_key, layer_index):
        """
        cells_display_key: string representing the cellDisplayWidget instance
        layer_index: Int between 0 and 6
        return pyqtgraph rois, original one or copies that are linked and non modifiables
        display_index 0 is the modifiable one
        """

        return self.rois_by_layer_dict[layer_index][cells_display_key]

    def remove_roi(self, roi_id, layer_index):
        """

        Args:
            roi_id:
            layer_index:

        Returns:

        """
        for cells_diplay_key, rois_list in self.rois_by_layer_dict[layer_index].items():
            new_rois = []
            for pg_roi in rois_list:
                if pg_roi.roi_id == roi_id:
                    # we need to remove it from the display
                    self.cells_display_dict[cells_diplay_key].remove_pg_roi(pg_roi)
                    continue
                new_rois.append(pg_roi)
            self.rois_by_layer_dict[layer_index][cells_diplay_key] = new_rois

    def add_pg_roi(self, contours, layer, add_to_cells_display=True):
        display_id = 0
        main_roi = PolyLineROI(contours, pen=(6, 9), closed=True, movable=True,
                               invisible_handle=False, alterable=True, no_seq_hover_action=False,
                               roi_id=self.individual_roi_id, layer_index=layer, roi_manager=self)
        if self.cells_display_keys[display_id] not in self.rois_by_layer_dict[layer]:
            self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]] = []
        self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]].append(main_roi)
        if add_to_cells_display:
            self.cells_display_dict[self.cells_display_keys[display_id]].add_pg_roi(main_roi)
        for display_id in np.arange(1, self.n_displays):
            other_roi = PolyLineROI(contours, pen=(6, 9), closed=True, movable=False,
                                    invisible_handle=False, alterable=False, no_seq_hover_action=True,
                                    roi_id=self.individual_roi_id, layer_index=layer, roi_manager=self)
            main_roi.link_a_roi(roi_to_link=other_roi)
            if self.cells_display_keys[display_id] not in self.rois_by_layer_dict[layer]:
                self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]] = []
            self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]].append(other_roi)
            if add_to_cells_display:
                self.cells_display_dict[self.cells_display_keys[display_id]].add_pg_roi(other_roi)
        self.individual_roi_id += 1

    def load_rois_coordinates_from_masks(self, mask_imgs):
        # rois c
        for layer, mask_img in enumerate(mask_imgs):

            self.rois_by_layer_dict[layer] = dict()
            contours = get_contours_from_mask_img(mask_img=mask_img)

            for contour in contours:
                display_id = 0
                main_roi = PolyLineROI(contour, pen=(6, 9), closed=True, movable=True,
                                       invisible_handle=False, alterable=True, no_seq_hover_action=False,
                                       roi_id=self.individual_roi_id, layer_index=layer, roi_manager=self)
                if self.cells_display_keys[display_id] not in self.rois_by_layer_dict[layer]:
                    self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]] = []
                self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]].append(main_roi)
                for display_id in np.arange(1, self.n_displays):
                    other_roi = PolyLineROI(contour, pen=(6, 9), closed=True, movable=False,
                                            invisible_handle=False, alterable=False, no_seq_hover_action=True,
                                            roi_id=self.individual_roi_id, layer_index=layer, roi_manager=self)
                    main_roi.link_a_roi(roi_to_link=other_roi)
                    if self.cells_display_keys[display_id] not in self.rois_by_layer_dict[layer]:
                        self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]] = []
                    self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]].append(other_roi)
                self.individual_roi_id += 1

            # now we want to create rois
            # print(f"contours len {len(contours)}")
            # print(f"contours {contours}")

    def _initiate_cells_id(self):
        # from laoded masks, determine how many cells are present and give an cell_id to each mask
        pass

    def load_pre_computed_coordinates(self, file_name):
        pass

    def fusion_rois(self):
        pass

    def get_cells_id(self):
        # return list of cell ids
        pass

    def get_cell_id_contours(self):
        # return a list of coords, as many as layers
        pass


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

    def keyPressEvent(self, event):
        """

        Args:
            event: Space: play from actual frame

        Returns:

        """
        # setting background picture
        # if event.key() == QtCore.Qt.Key_Space:
        #     if self.central_widget.playing:
        #         self.central_widget.pause()
        #     else:
        #         self.central_widget.start()
        # if event.key() == QtCore.Qt.Key_C:
        #     self.central_widget.set_current_timestep_to_actual_range()
        if event.key() == QtCore.Qt.Key_Plus or event.key() == QtCore.Qt.Key_Up:
            self.central_widget.change_layer(increment=True)
        if event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_Down:
            self.central_widget.change_layer(increment=False)
        # if event.key() == QtCore.Qt.Key_S:
        #     self.central_widget.ci_video_widget.switch_surprise_mfh()
        #     self.central_widget.behavior_video_widget_1.switch_surprise_mfh()
        #     self.central_widget.behavior_video_widget_2.switch_surprise_mfh()
        # if event.key() == QtCore.Qt.Key_Left:
        #     self.central_widget.previous_timestamp()
        # if event.key() == QtCore.Qt.Key_Right:
        #     self.central_widget.next_timestamp()


class MyQComboBox(QComboBox):
    """
    Special instance of ComboBox allowing to handle change so that it is connected to other combo_boxes
    """
    def __init__(self):
        """
        init
        """
        QComboBox.__init__(self)
        self.next_combo_box = None
        # each key represent a content to put in the list and the value could be either None, either
        #             another dict whose keys will be the content of the next ComboBox etc...
        self.choices_dict = None
        self.currentIndexChanged.connect(self.selection_change)

    def selection_change(self, index):
        """
        Called if the selection is changed either by the user or by the code
        Args:
            index:

        Returns:

        """
        # TODO: find data with multiple choices to test this code
        if self.next_combo_box is None:
            return

        # it should not be empty
        if self.count() == 0:
            return

        current_text = self.currentText()
        if current_text not in self.choices_dict:
            return

        content_next_combo_box = self.choices_dict[current_text]
        # removing previous items
        self.next_combo_box.clear()
        # adding new ones
        for choice_id in content_next_combo_box.keys():
            # need to put 2 arguments, in order to be able to find it using findData
            # self.next_combo_box.addItem(str(choice_id), str(choice_id))
            self.next_combo_box.addItem(get_icon_from_color("red"), str(choice_id))
        # to make combo_box following the next ones will be updated according to the content at the index 0
        self.next_combo_box.setCurrentIndex(0)


class MyQFrame(QFrame):

    def __init__(self, parent=None, with_description=True):
        """

        Args:
            analysis_arg:
            parent:
            with_description: if True, will add a description at the top of the widget
             based on a description arg if it exists
        """
        QFrame.__init__(self, parent=parent)

        self.description = ''
        self.long_description = None
        self.v_box = QVBoxLayout()

        self.h_box = QHBoxLayout()
        # if self.analysis_arg is not None:
        #     self.long_description = self.analysis_arg.get_long_description()

        self.q_label_empty = None
        # Trick to keep description in the middle even if help_button exists
        if with_description or (self.long_description is not None):
            self.q_label_empty = QLabel("  ")
            self.q_label_empty.setAlignment(Qt.AlignCenter)
            self.q_label_empty.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            self.q_label_empty.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            self.h_box.addWidget(self.q_label_empty)
            self.h_box.addStretch(1)

        # if with_description:
        #     if self.analysis_arg is not None:
        #         self.description = self.analysis_arg.get_short_description()
        #     if self.description:
        #
        #         self.q_label_description = QLabel(self.description)
        #         self.q_label_description.setAlignment(Qt.AlignCenter)
        #         self.q_label_description.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        #         self.q_label_description.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        #         self.h_box.addWidget(self.q_label_description)
        #     else:
        #         self.h_box.addStretch(1)
        if self.long_description:
            self.help_button = QPushButton()
            my_path = os.path.abspath(os.path.dirname(__file__))
            self.help_button.setIcon(QtGui.QIcon(os.path.join(my_path, 'icons/svg/question-mark.svg')))

            self.help_button.setIconSize(Core.QSize(10, 10))
            self.help_button.setToolTip(self.long_description)
            self.help_button.clicked.connect(self.help_click_event)

            self.h_box.addStretch(1)
            self.h_box.addWidget(self.help_button)
        elif self.q_label_empty is not None:
            self.h_box.addStretch(1)
            self.h_box.addWidget(self.q_label_empty)

        # TODO: See to remove one of the if
        if with_description or (self.long_description is not None):
            self.v_box.addLayout(self.h_box)

        self.v_box.addStretch(1)

        # if with_description or (self.long_description is not None):
        #     self.v_box.addLayout(self.h_box)

        self.setLayout(self.v_box)

        # if self.analysis_arg is not None:
        #     self.mandatory = self.analysis_arg.is_mandatory()
        # else:
        #     self.mandatory = False
        # self.setProperty("is_mandatory", str(self.mandatory))

    def change_mandatory_property(self, value):
        """
        Changing the property allowing to change the style sheet depending on the mandatory aspect of the argument
        Args:
            value:

        Returns:

        """
        self.setProperty("is_mandatory", value)
        self.style().unpolish(self)
        self.style().polish(self)

    def help_click_event(self):
        self.help_box = QMessageBox(self)
        my_path = os.path.abspath(os.path.dirname(__file__))
        self.help_box.setWindowIcon(QtGui.QIcon(os.path.join(my_path, 'icons/svg/cicada_open_focus.svg')))
        self.help_box.setIcon(QMessageBox.Information)
        if self.description:
            self.help_box.setWindowTitle(self.description)
        self.help_box.setAttribute(Qt.WA_DeleteOnClose)
        self.help_box.setStandardButtons(QMessageBox.Ok)
        self.help_box.setText(self.long_description)
        self.help_box.setModal(False)
        self.help_box.show()

    def get_layout(self):
        return self.v_box

    def set_property_to_missing(self):
        """
        Allows the change the stylesheet and indicate the user that a
        Returns:

        """
        self.setProperty("something_is_missing", "True")


class ComboBoxWidget(MyQFrame):

    def __init__(self, choices, ending_keys=None, horizontal_display=False, parent=None):
        """

        Args:
            analysis_arg: instance of AnalysisArgument
            parent:
        """
        MyQFrame.__init__(self, parent=parent)

        self.combo_boxes = dict()

        # represent the keys when to end the running down choices
        self.ending_keys = ending_keys

        # default_value = self.analysis_arg.get_default_value()
        # # legends: List of String, will be displayed as tooltip over the ComboBox
        # if hasattr(self.analysis_arg, "legends"):
        #     legends = self.analysis_arg.legends
        #     # if isinstance(legends, str):
        #     #     legends = [legends]
        # else:
        #     legends = None
        legends = None
        default_value = None

        # then each key represent a session_id and the value could be:
        # either a list of choices
        # either another dict, meaning will have more than one QCombotWidget
        index = 0

        self.combo_boxes = []
        self.add_multiple_combo_boxes(choices_dict=choices, legends=legends,
                                        index=0, ending_keys=ending_keys)
        if horizontal_display:
            h_box = QHBoxLayout()
        else:
            h_box = QVBoxLayout()
        # first we determine how many combo_box max
        n_boxes_max = 0
        v_box_session_id = QVBoxLayout()
        n_boxes_max = len(self.combo_boxes)
            # if len(self.combo_boxes) > 1:
            #     # if more than one session_id, we display the name of the session
            #     q_label = QLabel(session_id)
            #     # q_label.setAlignment(Qt.AlignCenter)
            #     q_label.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            #     q_label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
            #     v_box_session_id.addWidget(q_label)
        # if len(self.combo_boxes) > 1:
        #     h_box.addLayout(v_box_session_id)

        v_box_list = []
        for i in np.arange(n_boxes_max):
            v_box_list.append(QVBoxLayout())

        for index_combo, combo_box in enumerate(self.combo_boxes):
            v_box_list[index_combo].addWidget(combo_box)

        for v_box in v_box_list:
            h_box.addLayout(v_box)

        self.v_box.addLayout(h_box)
        self.v_box.addStretch(1)

        # is_mandatory = is_mandatory()
        # self.setProperty("is_mandatory", str(is_mandatory))

    def to_stretch(self):
        """
        Indicate if the widget should take all the space of a horizontal layout how might share the space
        with another widget
        Returns: Boolean

        """
        return True

    def add_multiple_combo_boxes(self,choices_dict, legends, index, ending_keys):
        """
        Allows to add multiple combo boxes, each changing the content of the next one for on given session_id
        Args:
            choices_dict: each key represent a content to put in the list and the value could be either None, either
            another dict which keys will be the content of the next ComboBox etc... or instead of a dict as value it
            could be a list that will define the content.
            legends:
            index:

        Returns:

        """
        combo_box = None

        index_loop_for = 0
        # combo_box following this one
        next_combo_box = None
        for choice_id, choice_content in choices_dict.items():
            if (ending_keys is not None) and (choice_id in ending_keys):
                continue
            if combo_box is None:
                combo_box = MyQComboBox()
                self.combo_boxes.append(combo_box)
            # need to put 2 arguments, in order to be able to find it using findData
            # combo_box.addItem(str(choice_id), str(choice_id))
            combo_box.addItem(get_icon_from_color("red"), str(choice_id))

            if choice_content is None:
                continue
            elif isinstance(choice_content, dict) and (index_loop_for == 0):
                next_combo_box = self.add_multiple_combo_boxes(choices_dict=choice_content,
                                                               legends=legends,
                                              index=index+1, ending_keys=ending_keys)
            elif isinstance(choice_content, list):
                next_combo_box = MyQComboBox()
                self.combo_boxes.append(next_combo_box)
                if legends is not None:
                    next_combo_box.setToolTip(legends[index+1])
                for next_choice_id in choice_content:
                    next_combo_box.addItem(str(next_choice_id), str(next_choice_id))

            index_loop_for += 1

        if combo_box is None:
            return None

        if legends is not None:
            if isinstance(legends, str):
                combo_box.setToolTip(legends)
            else:
                combo_box.setToolTip(legends[index])
        combo_box.choices_dict = choices_dict
        combo_box.next_combo_box = next_combo_box
        return combo_box

    def set_value(self, value):
        """
        Set a new value.
        Either value is None and nothing will happen
        If value is a list instance,
        Args:
            value:

        Returns:

        """
        if value is None:
            return

        if isinstance(value, dict):
            # means each key represent the session_id and the value the default value or values
            for session_id, value_to_set in value.items():
                # first checking is the session exists
                if session_id not in self.combo_boxes:
                    continue
                combo_box_list = self.combo_boxes[session_id]
                if not isinstance(value_to_set, list):
                    value_to_set = [value_to_set]
                if len(combo_box_list) != len(value_to_set):
                    # not compatible
                    continue
                for index_combo, combo_box in enumerate(combo_box_list):
                    index = combo_box.findData(value_to_set[index_combo])
                    # -1 for not found
                    if index != -1:
                        combo_box.setCurrentIndex(index)
        else:
            # otherwise we look for the value in each of the combo_box
            for combo_box_list in self.combo_boxes.values():
                if not isinstance(value, list):
                    value = [value]
                if len(combo_box_list) != len(value):
                    # not compatible
                    continue
                for index_combo, combo_box in enumerate(combo_box_list):
                    index = combo_box.findData(value[index_combo])
                    # -1 for not found
                    if index != -1:
                        combo_box.setCurrentIndex(index)

    def get_value(self):
        """

        Returns:

        """
        # if len(self.combo_boxes) == 1:
        #     for combo_box_list in self.combo_boxes.values():
        #         results = []
        #         for combo_box in combo_box_list:
        #             results.append(combo_box.currentText())
        #         if len(results) == 1:
        #             results = results[0]
        #         return results
        result_dict = dict()
        combo_box_list = self.combo_boxes
        results = []
        for combo_box in combo_box_list:
            results.append(combo_box.currentText())
        if len(results) == 1:
            results = results[0]
        return results


def get_icon_from_color(color):
    pixmap = QtGui.QPixmap(100, 100)
    pixmap.fill(QtGui.QColor(color))
    return QtGui.QIcon(pixmap)

"""
def get_icon_from_color(color):
    pixmap = QPixmap(100, 100)
    pixmap.fill(color)
    return QIcon(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = QComboBox()
    for text, color in (("item1", QColor("red")), ("item2", QColor(0xff00ff)), ("item3", QColor(0, 255, 0))):
        w.addItem(get_icon_from_color(color), text)
    w.show()
    sys.exit(app.exec_())
"""

class CentralWidget(QWidget):

    def __init__(self, main_window):
        super().__init__(parent=main_window)

        # root_path = "/Users/pappyhammer/Documents/academique/these_inmed/Lexi_Davide_project/"
        root_path = "/media/julien/Not_today/davide_lexi_project/11-2019 Davide - cfos/ANALYSIS/"

        result_path = os.path.join(root_path, "results_ld")

        mask_dir_path = os.path.join(root_path, "masques")
        red_dir_path = os.path.join(root_path, "cellules (red)")
        cfos_dir_path = os.path.join(root_path, "cfos (green)")

        self.images_dict = get_tiff_names(red_dir_path=red_dir_path, cfos_dir_path=cfos_dir_path, mask_dir_path=mask_dir_path)
        # raise Exception("KING IN THE NORTH")

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.rois_manager_dict = dict()

        self.current_layer = 0

        self.n_layers = 8

        self.main_layout = QHBoxLayout()

        self.grid_layout = QGridLayout()

        self.cells_widget = CellsDisplayMainWidget(current_z=self.current_layer, images_dict=self.images_dict,
                                                   key_image="red", central_widget=self,
                                                   id_widget="red", main_window=main_window)
        self.grid_layout.addWidget(self.cells_widget, 0, 0)

        self.cfos_widget = CellsDisplayMainWidget(current_z=self.current_layer, images_dict=self.images_dict,
                                                  key_image="cfos", central_widget=self,
                                              id_widget="cfos", main_window=main_window)
        self.grid_layout.addWidget(self.cfos_widget, 0, 1)

        self.cfos_widget.link_to_view(view=self.cells_widget.view)

        self.mask_widget = CellsDisplayMainWidget(current_z=self.current_layer, images_dict=self.images_dict,
                                                  key_image="mask", central_widget=self,
                                              id_widget="mask", main_window=main_window)
        self.grid_layout.addWidget(self.mask_widget, 1, 0)

        self.mask_widget.link_to_view(view=self.cells_widget.view)

        self.z_view_widget = ZViewWidget(n_layers=self.n_layers, current_layer=self.current_layer,
                                         main_window=main_window, width_image=500, parent=self)
        self.z_view_widget.link_to_view(view=self.cells_widget.view)
        self.grid_layout.addWidget(self.z_view_widget, 1, 1)
        self.z_view_widget.link_to_view(view=self.cells_widget.view)

        # self.overlap_widget = CellsDisplayMainWidget(current_z=self.current_layer, images_dict=self.images_dict,
        #                                         key_image="red",
        #                                       id_widget="red_bis", main_window=main_window)
        #
        # self.overlap_widget.link_to_view(view=self.cells_widget.view)
        # self.grid_layout.addWidget(self.overlap_widget, 1, 1)
        # , self.overlap_widget
        self.cells_display_widgets = [self.cells_widget, self.cfos_widget, self.mask_widget]
        self.cells_display_widgets_dict = {"red": self.cells_widget, "cfos": self.cfos_widget,
                                           "mask": self.mask_widget}

        self.main_layout.addLayout(self.grid_layout)

        self.control_panel_layout = QVBoxLayout()

        self.layer_layout = QHBoxLayout()
        self.layer_layout.addStretch(1)
        self.layer_label = QLabel("Layer")
        self.layer_spin_box = QSpinBox()
        self.layer_spin_box.setRange(0, 6)
        self.layer_spin_box.setSingleStep(1)
        self.layer_spin_box.setValue(0)
        # to just disable the text box but not the arrows
        self.layer_spin_box.lineEdit().setReadOnly(True)
        self.layer_spin_box.setToolTip("Layer")
        self.layer_spin_box.valueChanged.connect(self.layer_value_changed)
        self.layer_layout.addWidget(self.layer_label)
        self.layer_layout.addWidget(self.layer_spin_box)
        self.layer_layout.addStretch(1)
        self.control_panel_layout.addLayout(self.layer_layout)

        self.glue_layout = QHBoxLayout()
        self.combo_box_layout = QVBoxLayout()
        self.combo_box = ComboBoxWidget(choices=self.images_dict, ending_keys=["red", "cfos", "mask"],
                                        parent=self)
        self.combo_box_layout.addWidget(self.combo_box)
        self.display_button = QPushButton("Display", self)
        self.display_button.setToolTip("Display the selected field")
        self.display_button.clicked.connect(self.display_selected_field)
        self.combo_box_layout.addWidget(self.display_button)
        self.combo_box_layout.addStretch(1)

        self.glue_layout.addLayout(self.combo_box_layout)
        # self.glue_layout.addStretch(1)

        self.control_panel_layout.addLayout(self.glue_layout)

        self.main_layout.addLayout(self.control_panel_layout)

        self.setLayout(self.main_layout)

        # to display a first image
        self.display_selected_field()

    def layer_value_changed(self, value):
        """
            Called when self.layer_spin_box value is changed
            Returns:

        """
        self.current_layer = value
        for cells_display_widget in self.cells_display_widgets:
            cells_display_widget.set_layer(self.current_layer)
        # print(f"layer_value_changed {value}")

    def add_pg_roi(self, pos, image_keys):
        roi_manager = self.rois_manager_dict[tuple(image_keys)]
        size_half_square = 5
        contours = list()
        contours.append([pos[0]-size_half_square, pos[1]+size_half_square])
        contours.append([pos[0]+size_half_square, pos[1]+size_half_square])
        contours.append([pos[0]+size_half_square, pos[1]-size_half_square])
        contours.append([pos[0]-size_half_square, pos[1]-size_half_square])

        roi_manager.add_pg_roi(contours=contours, layer=self.current_layer)

    def change_layer(self, increment):
        """
                increment or decrement layer
                Args:
                    increment: bool

                Returns:

        """
        if increment:
            if self.current_layer + 1 > 6:
                return
            self.current_layer += 1
            self.layer_spin_box.setValue(self.current_layer)
        else:
            if self.current_layer - 1 < 0:
                return
            self.current_layer -= 1
            self.layer_spin_box.setValue(self.current_layer)

    def _get_rois_manager(self, image_keys):
        if image_keys not in self.rois_manager_dict:
            cells_display_keys = ["mask", "red", "cfos"]
            roi_manager = RoisManager(rois_manager_id=image_keys, n_displays=3,
                                      cells_display_keys=cells_display_keys,
                                      cells_display_dict=self.cells_display_widgets_dict)
            self.rois_manager_dict[image_keys] = roi_manager
            # if not yet created, then we load the rois from the mask data
            data_dict = get_data_in_dict_from_keys(list_keys=image_keys, data_dict=self.images_dict)
            mask_imgs = get_image_from_tiff(file_name=data_dict["mask"])
            roi_manager.load_rois_coordinates_from_masks(mask_imgs=mask_imgs)

        return self.rois_manager_dict[image_keys]

    def display_selected_field(self):
        image_keys = self.combo_box.get_value()
        roi_manager = self._get_rois_manager(tuple(image_keys))
        # print(f"image_keys {image_keys}")
        for cells_display_widget in self.cells_display_widgets:
            cells_display_widget.set_images(image_keys, roi_manager)


class MyViewBox(pg.ViewBox):
    """
    Mixed between RectMode and PanMode.
    Left click drag act like in RectMode
    Right click drag act life left click will act in PanMode (move the view box)
    Allow to zoom.
    Code from pyqtgraph examples
    """

    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        # self.setMouseMode(self.RectMode) ViewBox.PanMode

    def mouseClickEvent(self, ev):
        pass
        ## reimplement right-click to zoom out
        # if ev.button() == QtCore.Qt.RightButton:
        #     self.autoRange()

    def mouseDragEvent(self, ev):
        """
        Right click is used to zoom, left click is use to move the area
        Args:
            ev:

        Returns:

        """
        if ev.button() == QtCore.Qt.RightButton:
            self.setMouseMode(self.PanMode)
            # cheating, by telling it the left button is used instead
            ev._buttons = [QtCore.Qt.LeftButton]
            ev._button = QtCore.Qt.LeftButton
            pg.ViewBox.mouseDragEvent(self, ev)
        elif ev.button() == QtCore.Qt.LeftButton:
            self.setMouseMode(self.RectMode)
            pg.ViewBox.mouseDragEvent(self, ev)
        else:
            # ev.ignore()
            pg.ViewBox.mouseDragEvent(self, ev)


class ZViewWidget(pg.PlotWidget):

    def __init__(self, n_layers, current_layer,  main_window, width_image, parent):

        self.view_box = MyViewBox()

        pg.PlotWidget.__init__(self, parent=parent, viewBox=self.view_box)

        self.current_layer = current_layer
        self.n_layers = n_layers
        self.main_window = main_window

        self.pg_plot = self.getPlotItem()

        # self.pg_plot.hideAxis(axis='left')

        self.pg_plot.hideAxis(axis='bottom')

        self.view_box.setLimits(xMin=0, xMax=width_image, yMin=-1, yMax=self.n_layers)
        self.pg_plot.setXRange(0, width_image)
        self.pg_plot.setYRange(0, self.n_layers-1)

        self.pg_plot.setAspectLocked(False)

        color_pen = (0, 0, 255)
        self.current_layer_marker = pg.InfiniteLine(pos=[0, self.current_layer], angle=0,
                                                    pen=color_pen, movable=False,)
        self.pg_plot.addItem(item=self.current_layer_marker)

    def link_to_view(self, view):
        # TODO: see to fix it
        pass
        # self.view_box.setXLink(view=view)

class CellsDisplayMainWidget(pg.GraphicsLayoutWidget):
    """
    Module that will display the different w intervals along the frames
    """

    def __init__(self, id_widget, current_z, images_dict, key_image, main_window, central_widget, parent=None):

        # self.view_box = MyViewBox()
        pg.GraphicsLayoutWidget.__init__(self) # viewBox=self.view_box
        # allows the widget to be expanded in both axis
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.current_layer = current_z
        # see get_tiff_names for the structure
        # dict with: group, f, position, s, depth, key_image
        self.images_dict = images_dict
        self.id_widget = id_widget
        # ex: cfos, red, mask
        self.last_image_key = key_image
        self.main_window = main_window
        self.central_widget = central_widget
        # list of string representing the images actually displayed
        self.image_keys = None

        self.view = self.addViewBox(lockAspect=True, row=0, col=0, invertY=True, name=f"{id_widget}")
        # view.setMenuEnabled(False)

        self.image_displayed = pg.ImageItem(axisOrder='row-major', border='w')
        self.view.addItem(self.image_displayed)

        # different layer
        self.images = None
        # list of pyqtgraph rois
        self.current_pg_rois = []
        self.roi_manager = None

    def load_contours(self, contours):
        pass

    def set_images(self, image_keys, roi_manager):
        """
        image_keys: List of string
        """
        self.roi_manager = roi_manager
        self.image_keys = image_keys
        self._update_pg_rois()

        data_dict = get_data_in_dict_from_keys(list_keys=image_keys, data_dict=self.images_dict)
        self.images = get_image_from_tiff(file_name=data_dict[self.id_widget])
        # print(f"key_image {self.last_image_key} {data_dict[self.last_image_key]}")

        self._update_display()

    def remove_pg_roi(self, pg_roi):
        self.view.removeItem(pg_roi)

    def add_pg_roi(self, pg_roi):
        """
        Should be called from RoiManager, pg_roi should be registered in the RoiManager
        Args:
            pg_roi:

        Returns:

        """
        self.view.addItem(pg_roi)

    def _update_pg_rois(self):
        # pg as pyqtgraph
        new_pg_rois = self.roi_manager.get_pg_rois(cells_display_key=self.id_widget, layer_index=self.current_layer)
        if len(self.current_pg_rois) > 0:
            for pg_roi in self.current_pg_rois:
                self.view.removeItem(pg_roi)
        self.current_pg_rois = new_pg_rois
        for pg_roi in self.current_pg_rois:
            self.view.addItem(pg_roi)

    def set_layer(self, layer):
        self.current_layer = layer
        self._update_pg_rois()

        self._update_display()

    def _update_display(self):
        image_to_display = self.images[self.current_layer]
        if self.last_image_key == "mask":
            # otherwise the mask with no cell will be white instead of black
            if len(np.where(image_to_display == 0)[0]) > 4:
                image_to_display = np.invert(image_to_display)
            # print(
            #     f"image_to_display sum {np.sum(image_to_display)} {image_to_display.shape[0] * image_to_display.shape[1]} "
            #     f"min {np.min(image_to_display)}, max {np.max(image_to_display)}, "
            #     f"len(np.where(image_to_display == np.min(image_to_display)[0]) "
            #     f"{len(np.where(image_to_display == np.min(image_to_display))[0])}, "
            #     f"len(np.where(image_to_display == np.max(image_to_display)[0]) "
            #     f"{len(np.where(image_to_display == np.max(image_to_display))[0])}")

        #     image_to_display = np.reshape(image_to_display, (image_to_display.shape[0], image_to_display.shape[1], 1))
        # print(f"id_widget {self.id_widget}, image shape {image_to_display.shape}")
        self.image_displayed.setImage(image_to_display)

    def create_new_roi_in_the_middle(self):
        view_range = self.view.viewRange()
        # putting the new ROI in the middle
        pos = [np.mean(view_range[0]), np.mean(view_range[1])]
        self.central_widget.add_pg_roi(pos=pos, image_keys=self.image_keys)


    def keyPressEvent(self, event):
        """
        Call when a key is pressed
        Args:
            event:

        Returns:

        """
        if event.key() == QtCore.Qt.Key_R:
            self.create_new_roi_in_the_middle()
        # Sending the event to the main window if the widget is in the main window
        if self.main_window is not None:
            self.main_window.keyPressEvent(event=event)

    def link_to_view(self, view):
        self.view.setXLink(view=view)
        self.view.setYLink(view=view)


def get_data_in_dict_from_keys(list_keys, data_dict):
    if len(list_keys) > 0:
        return get_data_in_dict_from_keys(list_keys[1:], data_dict[list_keys[0]])
    return data_dict


def get_contours_from_mask_img(mask_img):
    """

    :param mask_img:
    :return: contours as a list of n_cells list, each following list contain pairs of int representing xy coords
    """
    mask_with_contours = mask_img.copy()
    if len(mask_with_contours.shape) < 3:
        mask_with_contours = np.reshape(mask_with_contours, (mask_with_contours.shape[0], mask_with_contours.shape[1], 1))
    # contours, hierarchy = cv2.findContours(mask_with_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(mask_with_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1

    coord_contours = []
    for contour in contours:
        xy = []
        for c in contour:
            xy.append([c[0][0]+0.5, c[0][1]+0.5])
        # removing the contour that take all the frame
        if [0.5, 0.5] in xy:
            continue
        coord_contours.append(xy)

    return coord_contours

class PlanMask:

    def __init__(self, mask_data, mask_id, result_path):
        self.mask_data = mask_data
        self.mask_id = mask_id
        self.result_path = result_path
        # plt.imshow(mask_data)
        # plt.show()
        mask_with_contours = mask_data.copy()
        mask_with_contours = np.reshape(mask_with_contours, (mask_with_contours.shape[0], mask_with_contours.shape[1], 1))
        # contours, hierarchy = cv2.findContours(mask_with_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #
        contours, hierarchy = cv2.findContours(mask_with_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.all_contours = contours
        # print(f"contours len {len(contours)}")
        # for contour in contours:
        #     print(f"contour shape {contour.shape}")
        # print(f"contours {contours}")
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
    root_path = "/media/julien/Not_today/davide_lexi_project/11-2019 Davide - cfos/ANALYSIS/"
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
    return get_image_from_tiff(file_name=file_name)

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
