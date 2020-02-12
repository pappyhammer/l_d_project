# from qtpy.QtWidgets import *
# from qtpy import QtGui
# from qtpy import QtCore
# from qtpy.QtCore import Qt
# import pyqtgraph as pg
# from PyQt5 import QtCore as Core
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
# from l_d_rois import PolyLineROI
import pickle
from shapely.geometry import Polygon, MultiPoint
import pandas as pd
from datetime import datetime

# TODO: ('GroupA', 'N1', 'ventr', 's1', 'dist')

BREWER_COLORS = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                 '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                 '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                 '#74add1', '#4575b4', '#313695']
DEFAULT_ROI_PEN_WIDTH = 10


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
            file_name_cropped = file_name[index_group + 1:]
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
            # if group == "GroupA" and f_value == "N1" and pos_value == "ventr":
            #     print(f"{[group, f_value, pos_value, s_value, depth_value, tiff_key]} {file_name}")
    return results_dict


# class MyQPushButton(QPushButton):
#
#     def __init__(self, cell_id, roi_manager):
#         self.roi_manager = roi_manager
#         self.cells_color = roi_manager.cells_color
#         self.cell_id = cell_id
#         QPushButton.__init__(self, str(cell_id))
#         self.setToolTip(f"Cell {cell_id}")
#         self.clicked.connect(self._button_action)
#         self.setStyleSheet(f"background-color:{self.cells_color[cell_id % len(self.cells_color)]}; color:black;")
#         self.activated = False
#
#     def change_activation_status(self):
#         if self.activated:
#             self.setStyleSheet(
#                 f"background-color:{self.cells_color[self.cell_id % len(self.cells_color)]}; color:black;")
#         else:
#             self.setStyleSheet(
#                 f"background-color:{self.cells_color[self.cell_id % len(self.cells_color)]}; color:white;")
#         self.activated = not self.activated
#
#     def indirect_button_action(self):
#         """
#         Means the user hasn't clicked on the button directly
#         Returns:
#
#         """
#         self._button_action()
#
#     def _button_action(self):
#         self.change_activation_status()
#         self.roi_manager.button_action(cell_id=self.cell_id)
#
#
# class RoisManager:
#
#     def __init__(self, rois_manager_id, n_displays, cells_display_keys, cells_display_dict, z_view_widget,
#                  images_dict,
#                  cells_buttons_layout, cells_n_layers_layout, n_layers=7):
#
#         self.n_layers = n_layers
#         # a ROI copied, in order to paste it later
#         self.copied_roi = None
#         # each roi has a cell_id
#         self.rois_by_layer_dict = dict()
#         self.z_view_widget = z_view_widget
#         # how many displays, one will have modifiable ROIs, the others will be link to the modifiable one
#         self.n_displays = n_displays
#         # rois_id: tuple of strings
#         self.rois_manager_id = rois_manager_id
#         # first one is for the red
#         self.cells_display_keys = cells_display_keys
#         # contains the instance of CellsDisplayMainWidget
#         self.cells_display_dict = cells_display_dict
#         # there because of PolyLineRoi instances
#         self.display_rois = True
#         # to keep them unique
#         self.individual_roi_id = 0
#         # dict with key a cell_id and with value a list of ROI_id (int)
#         self.cells_dict = dict()
#         # dict with key roi_id and return the pg_roi associated
#         self.pg_rois_dict = dict()
#         # up to 26 cells
#         self.cells_color = BREWER_COLORS
#
#         # see get_tiff_names for the structure
#         # dict with: group, f, position, s, depth, key_image
#         self.images_dict = images_dict
#
#         # for display purpose
#         self.cells_buttons_layout = cells_buttons_layout
#         self.cells_n_layers_layout = cells_n_layers_layout
#         # contains instance of QPushButton
#         self.cells_buttons_dict = SortedDict()
#         self.cell_n_layers_label_dict = SortedDict()
#
#         # cell id activated by a button
#         self.active_cell_id = None
#
#         # Incremented each time a new cell is created
#         self.cell_id = 0
#
#         # represent data saved than has been loaded
#         # Dict with key being the cell_id, and value is a dict with key the layer and then value a list of list of tuple
#         # of int representing the contours of each roi in the layer
#         self.pre_computed_data = None
#
#         # indicated that is pre_computed_data is available, if it has been used yet
#         self.pre_computed_data_not_loaded_yet = True
#
#         # represent the last roi for which the mouse went over
#         # tuple of 3 elemens (pg_roi, roi_id, layer_index)
#         self.last_roi_with_mouse_over = None
#
#     def is_pre_computed_data_available(self):
#         """
#         Return True if some saved_data (in a file) are available
#         :return:
#         """
#         return self.pre_computed_data is not None
#
#     def set_pre_computed_coordinates(self, data_dict):
#         """
#         Set pre-computed data.
#         Dict with key being the cell_id, and value is a dict with key the layer and then value a list of list of tuple
#         of int representing the contours of each roi in the layer
#         :param data_dict:
#         :return:
#         """
#         self.pre_computed_data = data_dict
#         self.pre_computed_data_not_loaded_yet = True
#
#     def load_saved_data(self):
#         """
#         Use pre-computed data previously saved
#         :return:
#         """
#         self.pre_computed_data_not_loaded_yet = False
#         self._erase_all()
#         for cell_id, layer_dict in self.pre_computed_data.items():
#             for layer, contours in layer_dict.items():
#                 for contour in contours:
#                     self.add_pg_roi(contours=contour, layer=layer, force_cell_id=cell_id, add_to_cells_display=False,
#                                     with_layout_update=False)
#             self._update_cell_label(cell_id=cell_id)
#         # self.update_buttons_layout()
#
#     def copy_roi(self, pg_roi):
#         """
#         Called when a ROI is copied through the action menu
#         :param pg_roi:
#         :return:
#         """
#         self.copied_roi = pg_roi
#
#     def paste_roi(self, layer):
#         """
#         Paste a copied roi to a given layer
#         :param layer:
#         :return:
#         """
#         if self.copied_roi is None:
#             return
#         # if self.copied_roi.layer_index == layer:
#         #     return
#         handle_name_positions = self.copied_roi.getLocalHandlePositions()
#         contours = [handle_name_pos[1] for handle_name_pos in handle_name_positions]
#         self.add_pg_roi(contours=contours, layer=layer)
#
#     def get_contours_data(self):
#         """
#         Get contours data a
#         Format is a dict, first key is the cell_id, value is a dict with key an int representing the layer then
#         the value is a list of list of tuples of 2 int representing x, y for each roi of the layer
#         :return:
#         """
#         # images_data_dict = get_data_in_dict_from_keys(list_keys=tuple(self.rois_manager_id), data_dict=self.images_dict)
#         # cfos_images = get_image_from_tiff(file_name=images_data_dict["cfos"])
#
#         data_dict = dict()
#         for cell_id, roi_ids in self.cells_dict.items():
#             if cell_id not in data_dict:
#                 data_dict[cell_id] = dict()
#
#             for roi_id in roi_ids:
#                 pg_roi = self.pg_rois_dict[roi_id]
#                 layer = pg_roi.layer_index
#
#                 if layer not in data_dict[cell_id]:
#                     data_dict[cell_id][layer] = []
#
#                 handle_name_positions = pg_roi.getLocalHandlePositions()
#                 # handle_name_pos[1] is an instance of QtCore.QPoint
#                 contours = [(handle_name_pos[1].x(), handle_name_pos[1].y()) for handle_name_pos in
#                             handle_name_positions]
#                 data_dict[cell_id][layer].append(contours)
#
#         return data_dict
#
#     def get_pg_rois(self, cells_display_key, layer_index):
#         """
#         cells_display_key: string representing the cellDisplayWidget instance
#         layer_index: Int between 0 and 6
#         return pyqtgraph rois, original one or copies that are linked and non modifiables
#         display_index 0 is the modifiable one
#         """
#
#         if layer_index not in self.rois_by_layer_dict:
#             return []
#         if cells_display_key not in self.rois_by_layer_dict[layer_index]:
#             return []
#         return self.rois_by_layer_dict[layer_index][cells_display_key]
#
#     def hover_event_on_roi(self, pg_roi):
#         """
#         Call when the mouse get hover a pg_roi
#         :param pg_roi:
#         :return:
#         """
#         roi_id = pg_roi.roi_id
#         layer_index = pg_roi.layer_index
#         self.last_roi_with_mouse_over = (pg_roi, roi_id, layer_index)
#         # to know if the hover is still on: pg_roi.mouseHovering == True ?
#
#     def _get_pg_rois_hovering(self):
#         """
#         Return a pg_roi if the mouse is hover one, None otherwise
#         :return:
#         """
#         if self.last_roi_with_mouse_over is None:
#             return None
#
#         if not self.last_roi_with_mouse_over[0].mouseHovering:
#             self.last_roi_with_mouse_over = None
#             return None
#
#         return self.last_roi_with_mouse_over[0]
#
#     def dilate_hover_roi(self):
#         """
#         Dilate the hover roi of x pixels
#         :return:
#         """
#
#         pg_roi = self._get_pg_rois_hovering()
#
#         if pg_roi is None:
#             return
#
#         layer_index = pg_roi.layer_index
#         roi_id = pg_roi.roi_id
#         cell_id = pg_roi.cell_id
#
#         if roi_id not in self.pg_rois_dict:
#             # meaning the roi doesn't exists anymore
#             return
#
#         # getting contours first
#         handle_name_positions = pg_roi.getLocalHandlePositions()
#         contours = [handle_name_pos[1] for handle_name_pos in handle_name_positions]
#
#         # then create shapely polygon
#         polygon = Polygon(contours)
#
#         # then dilate it
#         dilated_poly = polygon.buffer(0.5)
#         dilated_poly = dilated_poly.simplify(0.1, preserve_topology=False)
#         # then getting its contours
#         coords = np.array(dilated_poly.exterior.coords)
#         # print(f"coords {coords}")
#         # print(f"coords shape {coords.shape}")
#         # raise Exception("Dilatation over")
#
#         # then create a new roi
#         self.add_pg_roi(contours=coords, layer=layer_index, force_cell_id=cell_id)
#
#         # then delete the original roi
#         self.remove_roi(roi_id=roi_id, layer_index=layer_index)
#
#     def erode_hover_roi(self):
#         """
#         Erode the hover roi of x pixels
#         :return:
#         """
#
#         pg_roi = self._get_pg_rois_hovering()
#
#         if pg_roi is None:
#             return
#
#         layer_index = pg_roi.layer_index
#         roi_id = pg_roi.roi_id
#         cell_id = pg_roi.cell_id
#
#         if roi_id not in self.pg_rois_dict:
#             # meaning the roi doesn't exists anymore
#             return
#
#         # getting contours first
#         handle_name_positions = pg_roi.getLocalHandlePositions()
#         contours = [handle_name_pos[1] for handle_name_pos in handle_name_positions]
#
#         # then create shapely polygon
#         polygon = Polygon(contours)
#
#         # then dilate it
#         dilated_poly = polygon.buffer(-0.5)
#         dilated_poly = dilated_poly.simplify(0.1, preserve_topology=False)
#         # then getting its contours
#         coords = np.array(dilated_poly.exterior.coords)
#
#         # then create a new roi
#         self.add_pg_roi(contours=coords, layer=layer_index, force_cell_id=cell_id)
#
#         # then delete the original roi
#         self.remove_roi(roi_id=roi_id, layer_index=layer_index)
#
#     def copy_hover_roi(self):
#         """
#         Copy the hover roi
#         :return:
#         """
#         pg_roi = self._get_pg_rois_hovering()
#
#         if pg_roi is None:
#             # then we cancelled the last copied_roi
#             self.copied_roi = None
#             return
#
#         self.copy_roi(pg_roi=pg_roi)
#
#     def remove_hover_roi(self):
#         """
#         Remove the hover roi
#         :return:
#         """
#         pg_roi = self._get_pg_rois_hovering()
#
#         if pg_roi is None:
#             return
#
#         self.remove_roi(roi_id=pg_roi.roi_id, layer_index=pg_roi.layer_index)
#
#     def remove_roi(self, roi_id, layer_index):
#         """
#
#         Args:
#             roi_id:
#             layer_index:
#
#         Returns:
#
#         """
#         if roi_id not in self.pg_rois_dict:
#             return
#
#         del self.pg_rois_dict[roi_id]
#
#         for cells_diplay_key, rois_list in self.rois_by_layer_dict[layer_index].items():
#             new_rois = []
#             for pg_roi in rois_list:
#                 if pg_roi.roi_id == roi_id:
#                     # we need to remove it from the display
#                     self.cells_display_dict[cells_diplay_key].remove_pg_roi(pg_roi)
#                     if cells_diplay_key == "red":
#                         # main_roi
#                         # removing it from the cell_ids list
#                         cell_id = pg_roi.cell_id
#                         self.cells_dict[cell_id].remove(roi_id)
#                         if len(self.cells_dict[cell_id]) == 0:
#                             self._remove_cell(cell_id)
#                         else:
#                             self._update_cell_label(cell_id=cell_id)
#                     continue
#                 new_rois.append(pg_roi)
#             self.rois_by_layer_dict[layer_index][cells_diplay_key] = new_rois
#         self.z_view_widget.delete_associated_line(roi_id=roi_id)
#
#     def _remove_cell(self, cell_id):
#         """
#         Removing a given cell, no roi should be attached to it anymore
#         :param cell_id:
#         :return:
#         """
#         if len(self.cells_dict[cell_id]) > 0:
#             return
#
#         if self.active_cell_id == cell_id:
#             self.active_cell_id = None
#
#         del self.cells_dict[cell_id]
#
#         self._remove_button(cell_id)
#
#     def _remove_button(self, cell_id):
#         """
#         Remove the button associated to cell_id then update the buttons display
#         :param cell_id:
#         :return:
#         """
#         if cell_id not in self.cells_buttons_dict:
#             return
#
#         del self.cells_buttons_dict[cell_id]
#         del self.cell_n_layers_label_dict[cell_id]
#
#         self.update_buttons_layout()
#
#     def pg_roi_clicked(self, pg_roi):
#         """
#         Called when a ROI has been clicked on. Will change the roi cell_id if a button of the cell has been clicked
#         otherwise if no button is clicked, the cell of the ROI is now active (the button is activated)
#         :param pg_roi:
#         :return:
#         """
#         if self.active_cell_id is None:
#             button = self.cells_buttons_dict[pg_roi.cell_id]
#             button.indirect_button_action()
#             return
#         if self.active_cell_id == pg_roi.cell_id:
#             return
#
#         # we change the cell_id of the roi
#         old_cell_id = pg_roi.cell_id
#         pg_roi.cell_id = self.active_cell_id
#         for link_roi in pg_roi.linked_rois:
#             link_roi.cell_id = self.active_cell_id
#
#         self.cells_dict[old_cell_id].remove(pg_roi.roi_id)
#         self.cells_dict[self.active_cell_id].append(pg_roi.roi_id)
#         if len(self.cells_dict[old_cell_id]) == 0:
#             self._remove_cell(old_cell_id)
#         else:
#             self._update_cell_label(cell_id=old_cell_id)
#
#         self._update_cell_label(cell_id=self.active_cell_id)
#         self.update_colors()
#
#     def button_action(self, cell_id):
#         if self.active_cell_id is not None:
#             if self.active_cell_id == cell_id:
#                 self.active_cell_id = None
#             else:
#                 button = self.cells_buttons_dict[self.active_cell_id]
#                 button.change_activation_status()
#                 self.active_cell_id = cell_id
#         else:
#             self.active_cell_id = cell_id
#
#     def roi_updated(self, pg_roi):
#         """
#         Called when a roi has been updated (change of handle or moved)
#         If ROis has been removed, the method remove_roi() should be called instead
#         :param pg_roi:
#         :return:
#         """
#         if pg_roi.roi_id is None:
#             return
#         self.z_view_widget.update_associated_line(pg_roi=pg_roi, layer=None)
#
#     def are_rois_loaded(self):
#         if self.is_pre_computed_data_available() and self.pre_computed_data_not_loaded_yet:
#             return False
#         return len(self.rois_by_layer_dict) > 0
#
#     def add_pg_roi(self, contours, layer, add_to_cells_display=True, force_cell_id=None,
#                    with_layout_update=True):
#         """
#
#         :param contours:
#         :param layer:
#         :param add_to_cells_display:
#         :param force_cell_id: if not None, then the ROI will have as cell_id force_cell_id
#         :return:
#         """
#         # print(f"add_pg_roi contours {contours}")
#         display_id = 0
#         if force_cell_id is not None:
#             cell_id = force_cell_id
#         elif self.active_cell_id is None:
#             cell_id = self.cell_id
#             self.cell_id += 1
#         else:
#             # if a button is active, the ROI is part of the button cell
#             cell_id = self.active_cell_id
#         # white : (255, 255, 255)
#         roi_pen = pg.mkPen(color=self.cells_color[cell_id % len(self.cells_color)], width=DEFAULT_ROI_PEN_WIDTH)
#         main_roi = PolyLineROI(contours, pen=roi_pen, closed=True, movable=True,
#                                invisible_handle=False, alterable=True, no_seq_hover_action=False,
#                                roi_id=self.individual_roi_id, layer_index=layer, roi_manager=self)
#         self.pg_rois_dict[self.individual_roi_id] = main_roi
#         main_roi.cell_id = cell_id
#         if cell_id not in self.cells_dict:
#             self.cells_dict[cell_id] = []
#         self.cells_dict[cell_id].append(main_roi.roi_id)
#
#         # layer might not be created if data loaded from save_data
#         if layer not in self.rois_by_layer_dict:
#             self.rois_by_layer_dict[layer] = dict()
#
#         self.z_view_widget.update_associated_line(pg_roi=main_roi, layer=layer)
#         if self.cells_display_keys[display_id] not in self.rois_by_layer_dict[layer]:
#             self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]] = []
#         self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]].append(main_roi)
#         if add_to_cells_display:
#             self.cells_display_dict[self.cells_display_keys[display_id]].add_pg_roi(main_roi)
#         for display_id in np.arange(1, self.n_displays):
#             other_roi = PolyLineROI(contours, pen=roi_pen, closed=True, movable=False,
#                                     invisible_handle=True, alterable=False, no_seq_hover_action=True,
#                                     roi_id=self.individual_roi_id, layer_index=layer, roi_manager=self)
#             other_roi.cell_id = cell_id
#             main_roi.link_a_roi(roi_to_link=other_roi)
#             if self.cells_display_keys[display_id] not in self.rois_by_layer_dict[layer]:
#                 self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]] = []
#             self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]].append(other_roi)
#             if add_to_cells_display:
#                 self.cells_display_dict[self.cells_display_keys[display_id]].add_pg_roi(other_roi)
#         self.individual_roi_id += 1
#         if cell_id not in self.cells_buttons_dict:
#             self._add_cell_button(cell_id=cell_id, with_layout_update=with_layout_update)
#         else:
#             self._update_cell_label(cell_id=cell_id)
#
#     def load_rois_coordinates_from_masks(self, mask_imgs):
#         # rois c
#         for layer, mask_img in enumerate(mask_imgs):
#
#             self.rois_by_layer_dict[layer] = dict()
#             contours, centroids = get_contours_from_mask_img(mask_img=mask_img)
#
#             for contour_index, contour in enumerate(contours):
#                 display_id = 0
#                 main_roi = PolyLineROI(contour, pen=(6, 9), closed=True, movable=True,
#                                        invisible_handle=False, alterable=True, no_seq_hover_action=False,
#                                        roi_id=self.individual_roi_id, layer_index=layer, roi_manager=self,
#                                        original_centroid=centroids[contour_index])
#                 self.pg_rois_dict[self.individual_roi_id] = main_roi
#                 if self.cells_display_keys[display_id] not in self.rois_by_layer_dict[layer]:
#                     self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]] = []
#                 self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]].append(main_roi)
#                 for display_id in np.arange(1, self.n_displays):
#                     other_roi = PolyLineROI(contour, pen=(6, 9), closed=True, movable=False,
#                                             invisible_handle=True, alterable=False, no_seq_hover_action=True,
#                                             roi_id=self.individual_roi_id, layer_index=layer, roi_manager=self,
#                                             original_centroid=centroids[contour_index])
#                     main_roi.link_a_roi(roi_to_link=other_roi)
#                     if self.cells_display_keys[display_id] not in self.rois_by_layer_dict[layer]:
#                         self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]] = []
#                     self.rois_by_layer_dict[layer][self.cells_display_keys[display_id]].append(other_roi)
#                 self.individual_roi_id += 1
#         # Gathering ROI as cells
#         self._initiate_cells_id()
#
#     def _initiate_cells_id(self):
#         # from laoded masks, determine how many cells are present and give a cell_id to each mask
#         # going layer by layer, to each ROI that don't have a cell_id yet
#
#         # meaning that if two centroid from different layer are closer than 5 pixels, they are considered coming
#         # from the same cell
#         centroid_range = 10
#
#         main_cells_display_key = self.cells_display_keys[0]
#         for layer in range(self.n_layers):
#             if main_cells_display_key not in self.rois_by_layer_dict[layer]:
#                 # might happen if no ROI are registered
#                 continue
#             for pg_roi in self.rois_by_layer_dict[layer][main_cells_display_key]:
#                 if pg_roi.cell_id is not None:
#                     # cell_id already given
#                     continue
#                 pg_roi.cell_id = self.cell_id
#                 for link_roi in pg_roi.linked_rois:
#                     link_roi.cell_id = self.cell_id
#                 if self.cell_id not in self.cells_dict:
#                     self.cells_dict[self.cell_id] = []
#                 self.cells_dict[self.cell_id].append(pg_roi.roi_id)
#                 self.cell_id += 1
#                 centroid = np.array(pg_roi.original_centroid)
#                 # now looking in other layer for cell close by
#                 if layer < self.n_layers - 1:
#                     for other_layer in np.arange(layer + 1, self.n_layers):
#                         if main_cells_display_key not in self.rois_by_layer_dict[other_layer]:
#                             continue
#                         for other_pg_roi in self.rois_by_layer_dict[other_layer][main_cells_display_key]:
#                             if other_pg_roi.cell_id is not None:
#                                 continue
#                             other_centroid = np.array(other_pg_roi.original_centroid)
#                             dist = np.linalg.norm(centroid - other_centroid)
#                             if dist < centroid_range:
#                                 # then if the same cell
#                                 other_pg_roi.cell_id = pg_roi.cell_id
#                                 self.cells_dict[pg_roi.cell_id].append(other_pg_roi.roi_id)
#                                 for link_roi in other_pg_roi.linked_rois:
#                                     link_roi.cell_id = pg_roi.cell_id
#                             else:
#                                 # nothing, the id of the cell will be determined later on
#                                 pass
#         self._build_cells_buttons()
#
#     def update_colors(self):
#         # go over all ROIs (PolyLine and Line), and according to their cell_id, change their color
#         for layer in range(self.n_layers):
#             for display_key in self.rois_by_layer_dict[layer]:
#                 for pg_roi in self.rois_by_layer_dict[layer][display_key]:
#                     cell_id = pg_roi.cell_id
#                     color = self.cells_color[cell_id % len(self.cells_color)]
#                     roi_pen = pg.mkPen(color=color, width=DEFAULT_ROI_PEN_WIDTH)
#                     pg_roi.setPen(roi_pen)
#                     self.z_view_widget.update_line_color(pg_roi=pg_roi)
#
#     def _build_cells_buttons(self):
#         for cell_id in self.cells_dict.keys():
#             self._add_cell_button(cell_id, with_layout_update=False)
#
#     def _get_cell_n_layers(self, cell_id):
#         """
#         Give the number of layer with a given cell
#         :param cell_id:
#         :return:
#         """
#         if cell_id not in self.cells_dict:
#             return 0
#         roi_ids = self.cells_dict[cell_id]
#
#         n_layers = 0
#         for layer, rois_in_layer in self.rois_by_layer_dict.items():
#             if "red" not in rois_in_layer:
#                 continue
#             rois_ids_in_layer = [pg_roi.roi_id for pg_roi in rois_in_layer["red"]]
#             if len(np.intersect1d(roi_ids, rois_ids_in_layer)) > 0:
#                 n_layers += 1
#         return n_layers
#
#     def _get_number_of_rois_for_a_cell(self, cell_id):
#         if cell_id not in self.cells_dict:
#             return 0
#         return len(self.cells_dict[cell_id])
#
#     def _add_cell_button(self, cell_id, with_layout_update=True):
#         cell_button = MyQPushButton(cell_id=cell_id, roi_manager=self)
#
#         # self.cells_buttons_layout.addWidget(cell_button)
#         self.cells_buttons_dict[cell_id] = cell_button
#
#         cell_n_layers_label = QLabel()
#         cell_n_layers_label.setAlignment(Qt.AlignCenter)
#         cell_n_layers_label.setWindowFlags(QtCore.Qt.FramelessWindowHint)
#         cell_n_layers_label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
#         cell_n_layers_label.setToolTip(f"N layers & N ROIs for cell {cell_id}")
#         self.cell_n_layers_label_dict[cell_id] = cell_n_layers_label
#         self._update_cell_label(cell_id=cell_id)
#         if with_layout_update:
#             self.update_buttons_layout()
#
#     def _update_buttons_and_labels(self):
#         """
#         Used to re-create buttons after they have been deleted using deleteLater()
#         Returns:
#
#         """
#         cell_ids = list(self.cells_buttons_dict.keys())
#         for cell_id in cell_ids:
#             cell_button = MyQPushButton(cell_id=cell_id, roi_manager=self)
#             if self.active_cell_id is not None:
#                 if cell_id == self.active_cell_id:
#                     cell_button.change_activation_status()
#
#             # self.cells_buttons_layout.addWidget(cell_button)
#             self.cells_buttons_dict[cell_id] = cell_button
#
#             cell_n_layers_label = QLabel()
#             cell_n_layers_label.setAlignment(Qt.AlignCenter)
#             cell_n_layers_label.setWindowFlags(QtCore.Qt.FramelessWindowHint)
#             cell_n_layers_label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
#             cell_n_layers_label.setToolTip(f"N layers & N ROIs for cell {cell_id}")
#             self.cell_n_layers_label_dict[cell_id] = cell_n_layers_label
#             self._update_cell_label(cell_id=cell_id)
#
#         if self.active_cell_id is not None:
#             if self.active_cell_id not in cell_ids:
#                 # then no cell is active
#                 self.active_cell_id = None
#
#     def _update_cell_label(self, cell_id):
#         cell_n_layers_label = self.cell_n_layers_label_dict[cell_id]
#         n_layers = self._get_cell_n_layers(cell_id=cell_id)
#         n_rois = self._get_number_of_rois_for_a_cell(cell_id=cell_id)
#         cell_n_layers_label.setText(f"{n_layers} / {n_rois}")
#
#     def empty_buttons_layout(self):
#         for i in np.arange(self.cells_buttons_layout.count()):
#             item = self.cells_buttons_layout.itemAt(i)
#             item.widget().deleteLater()
#         # while self.cells_buttons_layout.count() > 0:
#         #     item = self.cells_buttons_layout.itemAt(0)
#         #     # adding the if to avoid this error
#         #     # QGraphicsScene::removeItem: item 0x7ffe4714f020's scene (0x0) is different from this scene (0x7ffe46c173d0)
#         #     # if item.scene() != 0:
#         #     self.cells_buttons_layout.removeWidget(item.widget())
#         #     # crash here
#         #     item.widget().deleteLater()
#         #     print("After deleteLater")
#         #     # item.widget().close()
#         # print(f"self.cells_buttons_layout.count() {self.cells_buttons_layout.count()}")
#         for i in np.arange(self.cells_n_layers_layout.count()):
#             item = self.cells_n_layers_layout.itemAt(i)
#             item.widget().deleteLater()
#         # while self.cells_n_layers_layout.count() > 0:
#         #     item = self.cells_n_layers_layout.itemAt(0)
#         #     # if item.widget().scene() != 0:
#         #     self.cells_n_layers_layout.removeWidget(item.widget())
#         #     item.widget().deleteLater()
#
#         # item.widget().close()
#
#     def update_buttons_layout(self):
#         # removing previous selection if there was one
#         # self.active_cell_id = None
#         self.empty_buttons_layout()
#         self._update_buttons_and_labels()
#         for cell_id, button in self.cells_buttons_dict.items():
#             button.show()
#             self.cells_buttons_layout.addWidget(button)
#
#         for cell_id, label in self.cell_n_layers_label_dict.items():
#             label.show()
#             self.cells_n_layers_layout.addWidget(label)
#
#     def _erase_all(self):
#         """
#         Erase all data in RoiManager.
#         Useful when we load data that has been saved
#         :return:
#         """
#         if len(self.pg_rois_dict) == 0:
#             return
#
#         copy_roi_dict = dict()
#         copy_roi_dict.update(self.pg_rois_dict)
#         for roi_id, pg_roi in copy_roi_dict.items():
#             layer = pg_roi.layer_index
#             self.remove_roi(roi_id=roi_id, layer_index=layer)
#         # in case some will still be there
#         self.empty_buttons_layout()
#
#     def fusion_rois(self):
#         pass
#
#     def get_cells_id(self):
#         # return list of cell ids
#         pass
#
#     def get_cell_id_contours(self):
#         # return a list of coords, as many as layers
#         pass
#
#
# class MainWindow(QMainWindow):
#     """Main window of the Exploratory GUI"""
#
#     def __init__(self, mask_dir_path, red_dir_path, cfos_dir_path):
#         super().__init__(parent=None)
#
#         self.setWindowTitle("cFos GUI")
#
#         screenGeometry = QApplication.desktop().screenGeometry()
#         # making sure the window is not bigger than the dimension of the screen
#         width_window = min(1800, screenGeometry.width())
#         # width_window = screenGeometry.width()
#         height_window = min(1000, screenGeometry.height())
#         self.resize(width_window, height_window)
#
#         ## creating widgets to put in the window
#         self.central_widget = CentralWidget(main_window=self, mask_dir_path=mask_dir_path,
#                                             red_dir_path=red_dir_path, cfos_dir_path=cfos_dir_path)
#         self.setCentralWidget(self.central_widget)
#
#         self.show()
#
#     def keyPressEvent(self, event):
#         """
#
#         Args:
#             event: Space: play from actual frame
#
#         Returns:
#
#         """
#         # setting background picture
#         # if event.key() == QtCore.Qt.Key_Space:
#         #     if self.central_widget.playing:
#         #         self.central_widget.pause()
#         #     else:
#         #         self.central_widget.start()
#         # if event.key() == QtCore.Qt.Key_C:
#         #     self.central_widget.set_current_timestep_to_actual_range()
#         if event.key() == QtCore.Qt.Key_Up:
#             self.central_widget.change_layer(increment=True)
#         if event.key() == QtCore.Qt.Key_Down:
#             self.central_widget.change_layer(increment=False)
#         for layer_index in range(self.central_widget.n_layers):
#             if event.key() == getattr(QtCore.Qt, f"Key_{layer_index}"):
#                 self.central_widget.change_layer(layer_index=layer_index)
#         if event.key() == QtCore.Qt.Key_D:
#             # display selected combo_boxes
#             self.central_widget.display_selected_field()
#         # if event.key() == QtCore.Qt.Key_S:
#         #     self.central_widget.ci_video_widget.switch_surprise_mfh()
#         #     self.central_widget.behavior_video_widget_1.switch_surprise_mfh()
#         #     self.central_widget.behavior_video_widget_2.switch_surprise_mfh()
#         # if event.key() == QtCore.Qt.Key_Left:
#         #     self.central_widget.previous_timestamp()
#         # if event.key() == QtCore.Qt.Key_Right:
#         #     self.central_widget.next_timestamp()
#
#
# class MyQComboBox(QComboBox):
#     """
#     Special instance of ComboBox allowing to handle change so that it is connected to other combo_boxes
#     """
#
#     def __init__(self, root_choices, status_color_fct):
#         """
#         init
#         """
#         QComboBox.__init__(self)
#         # represents all choices available
#         self.root_choices = root_choices
#         self.next_combo_box = None
#         self.previous_combo_box = None
#         self.status_color_fct = status_color_fct
#         # each key represent a content to put in the list and the value could be either None, either
#         #             another dict whose keys will be the content of the next ComboBox etc...
#         self.choices_dict = None
#         # fct that take as argument a list of string and return the status color for the combo_box
#         # self.update_combo_boxes_status = update_combo_boxes_status
#         # self.displayed_color_code = {True: "green", False: "red"}
#         self.currentIndexChanged.connect(self.selection_change)
#
#     def get_previous_combo_boxes_content(self, index):
#         """
#
#         :return: A list of string representing the content of the previous combo_boxes
#         """
#         if index == -1:
#             text_at_index = self.currentText()
#         else:
#             text_at_index = self.itemText(index)
#         if self.previous_combo_box is not None:
#             return self.previous_combo_box.get_previous_combo_boxes_content(index=-1) + [text_at_index]
#         return [text_at_index]
#
#     def selection_change(self, index):
#         """
#         Called if the selection is changed either by the user or by the code
#         Args:
#             index:
#
#         Returns:
#
#         """
#
#         if self.next_combo_box is None:
#             # # we change the color displayed according to the content of roi_manager
#             # image_keys = self.get_previous_combo_boxes_content()
#             # # has_been_displayed = self.roi_manager_dict[tuple(image_keys)].has_been_displayed
#             # current_index = self.currentIndex()
#             # self.setItemIcon(current_index, get_icon_from_color(self.status_color_fct(image_keys)))
#             # self.update_combo_boxes_status()
#             return
#
#         # it should not be empty
#         if self.count() == 0:
#             return
#
#         current_text = self.currentText()
#         if current_text not in self.choices_dict:
#             return
#
#         content_next_combo_box = self.choices_dict[current_text]
#         # removing previous items
#         self.next_combo_box.clear()
#         # adding new ones
#         for choice_id in content_next_combo_box.keys():
#             list_keys = self.get_previous_combo_boxes_content(self.currentIndex()) + [str(choice_id)]
#
#             # first we make sure this choice_id exists in the tree
#             if not check_if_list_of_keys_exists(list_keys=list_keys, my_dict=self.root_choices):
#                 print(f"Don't exist: {list_keys}")
#                 continue
#
#             # need to put 2 arguments, in order to be able to find it using findData
#             # self.next_combo_box.addItem(str(choice_id), str(choice_id))
#             status_color = self.status_color_fct(list_keys)
#             self.next_combo_box.addItem(get_icon_from_color(status_color), str(choice_id))
#         # to make combo_box following the next ones will be updated according to the content at the index 0
#         self.next_combo_box.setCurrentIndex(0)
#         # self.update_combo_boxes_status()


def check_if_list_of_keys_exists(list_keys, my_dict):
    """
    Return True if the list of keys allows to open succesive dictionnaries based on my_dict
    Args:
        list_keys: list of keys (string, int)
        my_dict:

    Returns:

    """
    if len(list_keys) == 0:
        return True

    if not isinstance(my_dict, SortedDict):
        return False

    if list_keys[0] in my_dict:
        return check_if_list_of_keys_exists(list_keys[1:], my_dict[list_keys[0]])
    return False

#
# class MyQFrame(QFrame):
#
#     def __init__(self, parent=None, with_description=True):
#         """
#
#         Args:
#             analysis_arg:
#             parent:
#             with_description: if True, will add a description at the top of the widget
#              based on a description arg if it exists
#         """
#         QFrame.__init__(self, parent=parent)
#
#         self.description = ''
#         self.long_description = None
#         self.v_box = QVBoxLayout()
#
#         self.h_box = QHBoxLayout()
#         # if self.analysis_arg is not None:
#         #     self.long_description = self.analysis_arg.get_long_description()
#
#         self.q_label_empty = None
#         # Trick to keep description in the middle even if help_button exists
#         if with_description or (self.long_description is not None):
#             self.q_label_empty = QLabel("  ")
#             self.q_label_empty.setAlignment(Qt.AlignCenter)
#             self.q_label_empty.setWindowFlags(QtCore.Qt.FramelessWindowHint)
#             self.q_label_empty.setAttribute(QtCore.Qt.WA_TranslucentBackground)
#             self.h_box.addWidget(self.q_label_empty)
#             self.h_box.addStretch(1)
#
#         # if with_description:
#         #     if self.analysis_arg is not None:
#         #         self.description = self.analysis_arg.get_short_description()
#         #     if self.description:
#         #
#         #         self.q_label_description = QLabel(self.description)
#         #         self.q_label_description.setAlignment(Qt.AlignCenter)
#         #         self.q_label_description.setWindowFlags(QtCore.Qt.FramelessWindowHint)
#         #         self.q_label_description.setAttribute(QtCore.Qt.WA_TranslucentBackground)
#         #         self.h_box.addWidget(self.q_label_description)
#         #     else:
#         #         self.h_box.addStretch(1)
#         if self.long_description:
#             self.help_button = QPushButton()
#             my_path = os.path.abspath(os.path.dirname(__file__))
#             self.help_button.setIcon(QtGui.QIcon(os.path.join(my_path, 'icons/svg/question-mark.svg')))
#
#             self.help_button.setIconSize(Core.QSize(10, 10))
#             self.help_button.setToolTip(self.long_description)
#             self.help_button.clicked.connect(self.help_click_event)
#
#             self.h_box.addStretch(1)
#             self.h_box.addWidget(self.help_button)
#         elif self.q_label_empty is not None:
#             self.h_box.addStretch(1)
#             self.h_box.addWidget(self.q_label_empty)
#
#         # TODO: See to remove one of the if
#         if with_description or (self.long_description is not None):
#             self.v_box.addLayout(self.h_box)
#
#         self.v_box.addStretch(1)
#
#         # if with_description or (self.long_description is not None):
#         #     self.v_box.addLayout(self.h_box)
#
#         self.setLayout(self.v_box)
#
#         # if self.analysis_arg is not None:
#         #     self.mandatory = self.analysis_arg.is_mandatory()
#         # else:
#         #     self.mandatory = False
#         # self.setProperty("is_mandatory", str(self.mandatory))
#
#     def change_mandatory_property(self, value):
#         """
#         Changing the property allowing to change the style sheet depending on the mandatory aspect of the argument
#         Args:
#             value:
#
#         Returns:
#
#         """
#         self.setProperty("is_mandatory", value)
#         self.style().unpolish(self)
#         self.style().polish(self)
#
#     def help_click_event(self):
#         self.help_box = QMessageBox(self)
#         my_path = os.path.abspath(os.path.dirname(__file__))
#         self.help_box.setWindowIcon(QtGui.QIcon(os.path.join(my_path, 'icons/svg/cicada_open_focus.svg')))
#         self.help_box.setIcon(QMessageBox.Information)
#         if self.description:
#             self.help_box.setWindowTitle(self.description)
#         self.help_box.setAttribute(Qt.WA_DeleteOnClose)
#         self.help_box.setStandardButtons(QMessageBox.Ok)
#         self.help_box.setText(self.long_description)
#         self.help_box.setModal(False)
#         self.help_box.show()
#
#     def get_layout(self):
#         return self.v_box
#
#     def set_property_to_missing(self):
#         """
#         Allows the change the stylesheet and indicate the user that a
#         Returns:
#
#         """
#         self.setProperty("something_is_missing", "True")
#
#
# class ComboBoxWidget(MyQFrame):
#
#     def __init__(self, choices, status_color_fct,
#                  ending_keys=None, horizontal_display=False, parent=None):
#         """
#
#         Args:
#             analysis_arg: instance of AnalysisArgument
#             parent:
#         """
#         MyQFrame.__init__(self, parent=parent)
#
#         self.original_choices = choices
#         self.combo_boxes = dict()
#         self.status_color_fct = status_color_fct
#
#         # represent the keys when to end the running down choices
#         self.ending_keys = ending_keys
#
#         # default_value = self.analysis_arg.get_default_value()
#         # # legends: List of String, will be displayed as tooltip over the ComboBox
#         # if hasattr(self.analysis_arg, "legends"):
#         #     legends = self.analysis_arg.legends
#         #     # if isinstance(legends, str):
#         #     #     legends = [legends]
#         # else:
#         #     legends = None
#         legends = None
#         default_value = None
#
#         # then each key represent a session_id and the value could be:
#         # either a list of choices
#         # either another dict, meaning will have more than one QCombotWidget
#         index = 0
#
#         self.combo_boxes = []
#         self.add_multiple_combo_boxes(choices_dict=choices, legends=legends,
#                                       index=0, ending_keys=ending_keys)
#         if horizontal_display:
#             h_box = QHBoxLayout()
#         else:
#             h_box = QVBoxLayout()
#         # first we determine how many combo_box max
#         n_boxes_max = 0
#         v_box_session_id = QVBoxLayout()
#         n_boxes_max = len(self.combo_boxes)
#         # if len(self.combo_boxes) > 1:
#         #     # if more than one session_id, we display the name of the session
#         #     q_label = QLabel(session_id)
#         #     # q_label.setAlignment(Qt.AlignCenter)
#         #     q_label.setWindowFlags(QtCore.Qt.FramelessWindowHint)
#         #     q_label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
#         #     v_box_session_id.addWidget(q_label)
#         # if len(self.combo_boxes) > 1:
#         #     h_box.addLayout(v_box_session_id)
#
#         v_box_list = []
#         for i in np.arange(n_boxes_max):
#             v_box_list.append(QVBoxLayout())
#
#         for index_combo, combo_box in enumerate(self.combo_boxes):
#             v_box_list[index_combo].addWidget(combo_box)
#
#         for v_box in v_box_list:
#             h_box.addLayout(v_box)
#
#         self.v_box.addLayout(h_box)
#         self.v_box.addStretch(1)
#
#         # is_mandatory = is_mandatory()
#         # self.setProperty("is_mandatory", str(is_mandatory))
#
#     def to_stretch(self):
#         """
#         Indicate if the widget should take all the space of a horizontal layout how might share the space
#         with another widget
#         Returns: Boolean
#
#         """
#         return True
#
#     def update_combo_boxes_status(self):
#         """
#
#         :return:
#         """
#         # goes through all the combo_boxes, and update the item status color
#         for combo_box in self.combo_boxes:
#             for index_combo_box in range(combo_box.count()):
#                 image_keys = combo_box.get_previous_combo_boxes_content(index_combo_box)
#
#                 status_color = self.status_color_fct(image_keys)
#                 combo_box.setItemIcon(index_combo_box, get_icon_from_color(status_color))
#
#     def add_multiple_combo_boxes(self, choices_dict, legends, index, ending_keys):
#         """
#         Allows to add multiple combo boxes, each changing the content of the next one for on given session_id
#         Args:
#             choices_dict: each key represent a content to put in the list and the value could be either None, either
#             another dict which keys will be the content of the next ComboBox etc... or instead of a dict as value it
#             could be a list that will define the content.
#             legends:
#             index:
#
#         Returns:
#
#         """
#         combo_box = None
#
#         index_loop_for = 0
#         # combo_box following this one
#         next_combo_box = None
#         for choice_id, choice_content in choices_dict.items():
#             if (ending_keys is not None) and (choice_id in ending_keys):
#                 continue
#             if combo_box is None:
#                 combo_box = MyQComboBox(root_choices=self.original_choices, status_color_fct=self.status_color_fct)
#                 self.combo_boxes.append(combo_box)
#             combo_box.addItem(get_icon_from_color("red"), str(choice_id))
#
#             if choice_content is None:
#                 continue
#             elif isinstance(choice_content, dict) and (index_loop_for == 0):
#                 if choice_id in choice_content:
#                     # to solve a bug i don't understand
#                     choice_content = choice_content[choice_id]
#                 next_combo_box_tmp = self.add_multiple_combo_boxes(choices_dict=choice_content,
#                                                                    legends=legends,
#                                                                    index=index + 1, ending_keys=ending_keys)
#                 if next_combo_box is None:
#                     next_combo_box = next_combo_box_tmp
#             # elif isinstance(choice_content, list):
#             #     next_combo_box = MyQComboBox()
#             #     self.combo_boxes.append(next_combo_box)
#             #     if legends is not None:
#             #         next_combo_box.setToolTip(legends[index+1])
#             #     for next_choice_id in choice_content:
#             #         next_combo_box.addItem(str(next_choice_id), str(next_choice_id))
#
#             index_loop_for += 1
#
#         if combo_box is None:
#             return None
#
#         if legends is not None:
#             if isinstance(legends, str):
#                 combo_box.setToolTip(legends)
#             else:
#                 combo_box.setToolTip(legends[index])
#
#         combo_box.choices_dict = choices_dict
#         combo_box.next_combo_box = next_combo_box
#         if next_combo_box is not None:
#             next_combo_box.previous_combo_box = combo_box
#         return combo_box
#
#     def set_value(self, value):
#         """
#         Set a new value.
#         Either value is None and nothing will happen
#         If value is a list instance,
#         Args:
#             value:
#
#         Returns:
#
#         """
#         if value is None:
#             return
#
#         if isinstance(value, dict):
#             # means each key represent the session_id and the value the default value or values
#             for session_id, value_to_set in value.items():
#                 # first checking is the session exists
#                 if session_id not in self.combo_boxes:
#                     continue
#                 combo_box_list = self.combo_boxes[session_id]
#                 if not isinstance(value_to_set, list):
#                     value_to_set = [value_to_set]
#                 if len(combo_box_list) != len(value_to_set):
#                     # not compatible
#                     continue
#                 for index_combo, combo_box in enumerate(combo_box_list):
#                     index = combo_box.findData(value_to_set[index_combo])
#                     # -1 for not found
#                     if index != -1:
#                         combo_box.setCurrentIndex(index)
#         else:
#             # otherwise we look for the value in each of the combo_box
#             for combo_box_list in self.combo_boxes.values():
#                 if not isinstance(value, list):
#                     value = [value]
#                 if len(combo_box_list) != len(value):
#                     # not compatible
#                     continue
#                 for index_combo, combo_box in enumerate(combo_box_list):
#                     index = combo_box.findData(value[index_combo])
#                     # -1 for not found
#                     if index != -1:
#                         combo_box.setCurrentIndex(index)
#
#     def get_value(self):
#         """
#
#         Returns:
#
#         """
#         # if len(self.combo_boxes) == 1:
#         #     for combo_box_list in self.combo_boxes.values():
#         #         results = []
#         #         for combo_box in combo_box_list:
#         #             results.append(combo_box.currentText())
#         #         if len(results) == 1:
#         #             results = results[0]
#         #         return results
#         result_dict = dict()
#         combo_box_list = self.combo_boxes
#         results = []
#         for combo_box in combo_box_list:
#             results.append(combo_box.currentText())
#         if len(results) == 1:
#             results = results[0]
#         return results
#
#
# def get_icon_from_color(color):
#     pixmap = QtGui.QPixmap(100, 100)
#     pixmap.fill(QtGui.QColor(color))
#     return QtGui.QIcon(pixmap)


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


def get_tree_dict_as_a_list(tree_dict):
    """
    Take a dict that contains as value only other dicts or list of simple types value sor a simple type value
    and return a list of list of all keys with last data at the end
    Args:
        tree_dict:

    Returns:

    """
    tree_as_list = []
    for key, sub_tree in tree_dict.items():
        # branch_list = [key]
        # tree_as_list.append(branch_list)
        if isinstance(sub_tree, dict):
            branches = get_tree_dict_as_a_list(tree_dict=sub_tree)
            tree_as_list.extend([[key] + branch for branch in branches])
        elif isinstance(sub_tree, list):
            # means we reached a leaves
            for leaf in sub_tree:
                tree_as_list.append([key, leaf])
        else:
            # means we reached a leaf
            leaf = sub_tree
            tree_as_list.append([key, leaf])
    return tree_as_list


# class CentralWidget(QWidget):
#
#     def __init__(self, main_window, mask_dir_path, red_dir_path, cfos_dir_path):
#         super().__init__(parent=main_window)
#
#         self.images_dict = get_tiff_names(red_dir_path=red_dir_path, cfos_dir_path=cfos_dir_path,
#                                           mask_dir_path=mask_dir_path)
#
#         self.all_image_keys = get_tree_dict_as_a_list(self.images_dict)
#         # print(f"self.all_image_keys")
#         # for image_keys in self.all_image_keys:
#         #     print(f"{image_keys}")
#
#         # removing the two last keys which are like "mask", "red" and the tiffs file_name
#         self.all_image_keys = set([tuple(images[:-2]) for images in self.all_image_keys])
#         # raise Exception("KING IN THE NORTH")
#
#         # Enable antialiasing for prettier plots
#         pg.setConfigOptions(antialias=True)
#
#         # we have on roi_manager for each image_keys, the key is a tuple of string representing the image
#         self.rois_manager_dict = dict()
#
#         # we have on data loaded for each image_keys, the key is a tuple of string representing the image
#         self.loaded_data_dict = dict()
#
#         self.current_layer = 0
#
#         self.n_layers = 7
#
#         self.cells_buttons_layout = QVBoxLayout()
#
#         self.cells_n_layers_layout = QVBoxLayout()
#
#         self.main_layout = QHBoxLayout()
#
#         self.grid_layout = QGridLayout()
#
#         self.cells_widget = CellsDisplayMainWidget(current_z=self.current_layer, images_dict=self.images_dict,
#                                                    key_image="red", central_widget=self,
#                                                    id_widget="red", main_window=main_window)
#         self.grid_layout.addWidget(self.cells_widget, 0, 0)
#
#         self.cfos_widget = CellsDisplayMainWidget(current_z=self.current_layer, images_dict=self.images_dict,
#                                                   key_image="cfos", central_widget=self,
#                                                   id_widget="cfos", main_window=main_window)
#         self.grid_layout.addWidget(self.cfos_widget, 0, 1)
#
#         self.cfos_widget.link_to_view(view=self.cells_widget.view)
#
#         self.mask_widget = CellsDisplayMainWidget(current_z=self.current_layer, images_dict=self.images_dict,
#                                                   key_image="mask", central_widget=self,
#                                                   id_widget="mask", main_window=main_window)
#         self.grid_layout.addWidget(self.mask_widget, 1, 0)
#
#         self.mask_widget.link_to_view(view=self.cells_widget.view)
#
#         self.z_view_widget = ZViewWidget(n_layers=self.n_layers, current_layer=self.current_layer,
#                                          images_dict=self.images_dict,
#                                          main_window=main_window, width_image=500, parent=self,
#                                          linked_cells_display_widget=self.cells_widget)
#         self.z_view_widget.link_to_view(view_to_link=self.cells_widget.view)
#         self.grid_layout.addWidget(self.z_view_widget, 1, 1)
#         # self.cells_widget.view.sigXRangeChanged.connect(self.z_view_widget.update_region)
#
#         # self.overlap_widget = CellsDisplayMainWidget(current_z=self.current_layer, images_dict=self.images_dict,
#         #                                         key_image="red",
#         #                                       id_widget="red_bis", main_window=main_window)
#         #
#         # self.overlap_widget.link_to_view(view=self.cells_widget.view)
#         # self.grid_layout.addWidget(self.overlap_widget, 1, 1)
#         # , self.overlap_widget
#         self.cells_display_widgets = [self.cells_widget, self.cfos_widget, self.mask_widget, self.z_view_widget]
#         self.cells_display_widgets_dict = {"red": self.cells_widget, "cfos": self.cfos_widget,
#                                            "mask": self.mask_widget}
#
#         self.main_layout.addLayout(self.grid_layout)
#
#         # Creating the roi managers
#         # Use saved rois to load them
#         cells_display_keys = ["red", "mask", "cfos"]
#         for image_keys in self.all_image_keys:
#             if image_keys in self.rois_manager_dict:
#                 continue
#             roi_manager = RoisManager(rois_manager_id=image_keys, n_displays=3,
#                                       z_view_widget=self.z_view_widget,
#                                       images_dict=self.images_dict,
#                                       cells_display_keys=cells_display_keys,
#                                       cells_buttons_layout=self.cells_buttons_layout,
#                                       cells_n_layers_layout=self.cells_n_layers_layout,
#                                       cells_display_dict=self.cells_display_widgets_dict)
#             self.rois_manager_dict[image_keys] = roi_manager
#
#         self.control_panel_layout = QVBoxLayout()
#
#         self.layer_layout = QHBoxLayout()
#         self.layer_layout.addStretch(1)
#         self.layer_label = QLabel("Layer")
#         self.layer_spin_box = QSpinBox()
#         self.layer_spin_box.setRange(0, 6)
#         self.layer_spin_box.setSingleStep(1)
#         self.layer_spin_box.setValue(0)
#         # to just disable the text box but not the arrows
#         self.layer_spin_box.lineEdit().setReadOnly(True)
#         self.layer_spin_box.setToolTip("Layer")
#         self.layer_spin_box.valueChanged.connect(self.layer_value_changed)
#         self.layer_layout.addWidget(self.layer_label)
#         self.layer_layout.addWidget(self.layer_spin_box)
#         self.layer_layout.addStretch(1)
#         self.control_panel_layout.addLayout(self.layer_layout)
#
#         self.glue_layout = QHBoxLayout()
#         self.combo_box_layout = QVBoxLayout()
#         self.combo_box = ComboBoxWidget(choices=self.images_dict,
#                                         status_color_fct=self.status_color_fct,
#                                         ending_keys=["red", "cfos", "mask"],
#                                         parent=self)
#         self.combo_box_layout.addWidget(self.combo_box)
#         self.display_button = QPushButton("Display", self)
#         self.display_button.setToolTip("Display the selected field")
#         self.display_button.clicked.connect(self.display_selected_field)
#         self.combo_box_layout.addWidget(self.display_button)
#
#         self.save_rois_button = QPushButton("Save ROIs", self)
#         self.save_rois_button.setToolTip("Save ROIs that has been checked in a file")
#         self.save_rois_button.clicked.connect(self.save_rois)
#         self.combo_box_layout.addWidget(self.save_rois_button)
#
#         self.load_rois_button = QPushButton("Load ROIs", self)
#         self.load_rois_button.setToolTip("Load ROIs from file")
#         self.load_rois_button.clicked.connect(self.load_rois)
#         self.combo_box_layout.addWidget(self.load_rois_button)
#
#         button_info_layout = QHBoxLayout()
#         button_info_layout.addLayout(self.cells_buttons_layout)
#         button_info_layout.addLayout(self.cells_n_layers_layout)
#         self.combo_box_layout.addLayout(button_info_layout)
#         self.combo_box_layout.addStretch(1)
#
#         self.glue_layout.addLayout(self.combo_box_layout)
#         # self.glue_layout.addStretch(1)
#
#         self.control_panel_layout.addLayout(self.glue_layout)
#
#         self.main_layout.addLayout(self.control_panel_layout)
#
#         self.setLayout(self.main_layout)
#
#         self.displayed_image_keys = None
#         # to display a first image
#         self.display_selected_field()
#
#     def update_cells_buttons_layout(self):
#         """
#         Update the Cells buttons layout displayed depending on the images displayed
#         :return:
#         """
#         roi_manager = self.rois_manager_dict[tuple(self.displayed_image_keys)]
#         roi_manager.update_buttons_layout()
#
#     def save_rois(self):
#         """
#         Save rois in a file, save only the rois from images than have been loaded (aka checked)
#         :return:
#         """
#         file_dialog = QFileDialog(self, "Saving ROIs")
#
#         # setting options
#         options = QFileDialog.Options()
#         options |= QFileDialog.DontUseNativeDialog
#         options |= QFileDialog.DontUseCustomDirectoryIcons
#         file_dialog.setOptions(options)
#
#         # ARE WE TALKING ABOUT FILES OR FOLDERS
#         file_dialog.setFileMode(QFileDialog.AnyFile)
#         file_dialog.setNameFilter("Pickle files (*.pkl)")
#         # file_dialog.setNameFilter("Npz files (*.npz)")
#
#         # OPENING OR SAVING
#         file_dialog.setAcceptMode(QFileDialog.AcceptSave)
#
#         # SET THE STARTING DIRECTORY
#         # default_value = self.analysis_arg.get_default_value()
#         # if default_value is not None and isinstance(default_value, str):
#         #     self.file_dialog.setDirectory(default_value)
#         if file_dialog.exec_() == QDialog.Accepted:
#             file_name = file_dialog.selectedFiles()[0]
#         else:
#             return
#
#         data_to_save = dict()
#         for image_keys in self.all_image_keys:
#             if image_keys not in self.rois_manager_dict:
#                 continue
#             roi_manager = self.rois_manager_dict[image_keys]
#             if not roi_manager.are_rois_loaded() and (image_keys not in self.loaded_data_dict):
#                 # if contours have not been loaded or modified, we don't save them
#                 continue
#             if roi_manager.are_rois_loaded():
#                 # getting data from roi_manager
#                 data_to_save[image_keys] = roi_manager.get_contours_data()
#             else:
#                 # getting data from loaded one, not yet activated in the RoiManager
#                 data_to_save[image_keys] = self.loaded_data_dict[image_keys]
#         # print(f"data_to_save {data_to_save}")
#         with open(file_name, 'wb') as f:
#             pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)
#
#     def load_rois(self):
#         """
#         Load rois from file, replace the existing rois for the images corresponding
#         :return:
#         """
#         file_dialog = QFileDialog(self, "Loading ROIs")
#
#         # setting options
#         options = QFileDialog.Options()
#         options |= QFileDialog.DontUseNativeDialog
#         options |= QFileDialog.DontUseCustomDirectoryIcons
#         file_dialog.setOptions(options)
#
#         # ARE WE TALKING ABOUT FILES OR FOLDERS
#         file_dialog.setFileMode(QFileDialog.ExistingFiles)
#         file_dialog.setNameFilter("Pickle files (*.pkl)")
#
#         # OPENING OR SAVING
#         file_dialog.setAcceptMode(QFileDialog.AcceptOpen)
#
#         # SET THE STARTING DIRECTORY
#         # default_value = self.analysis_arg.get_default_value()
#         # if default_value is not None and isinstance(default_value, str):
#         #     self.file_dialog.setDirectory(default_value)
#
#         # print(f"if file_dialog.exec_() == QDialog.Accepted")
#         # print(f"file_dialog.exec_() {file_dialog.exec_()}")
#         if file_dialog.exec() != QDialog.Accepted:
#             return
#
#         file_name = file_dialog.selectedFiles()[0]
#         with open(file_name, 'rb') as f:
#             pickle_data = pickle.load(f)
#             print(f"N images in pickle file: {len(pickle_data)}")
#             list_to_display = []
#             for key, value in pickle_data.items():
#                 list_to_display.append(key)
#                 # print(f"{key}")
#             list_to_display.sort()
#             for item in list_to_display:
#                 print(item)
#             print("")
#             print(f"N images already loaded: {len(self.loaded_data_dict)}")
#             list_to_display = []
#             for key, value in self.loaded_data_dict.items():
#                 list_to_display.append(key)
#             list_to_display.sort()
#             for item in list_to_display:
#                 print(item)
#             size_loaded_dict_before_update = len(self.loaded_data_dict)
#             print("")
#             self.loaded_data_dict.update(pickle_data)
#
#             if size_loaded_dict_before_update > 0:
#                 print(f"N images now loaded: {len(self.loaded_data_dict)}")
#                 list_to_display = []
#                 for key, value in self.loaded_data_dict.items():
#                     list_to_display.append(key)
#                 list_to_display.sort()
#                 for item in list_to_display:
#                     print(item)
#                 print("")
#             self._update_data_from_loaded_one(data_loaded_dict=pickle_data)
#
#     def _update_data_from_loaded_one(self, data_loaded_dict):
#         """
#         Use information in self.loaded_data_dict to update the display or ROIs and combo_boxes
#         :return:
#         """
#         for image_keys, data_dict in data_loaded_dict.items():
#             if image_keys in self.rois_manager_dict:
#                 # updating roi_manager (will update the buttons and the images display)
#                 roi_manager = self.rois_manager_dict[image_keys]
#                 roi_manager.set_pre_computed_coordinates(data_dict=data_dict)
#                 # updating the combo_box
#         # updating the current selected combo_box if loaded data is now available
#         self.displayed_image_keys = self.combo_box.get_value()
#         roi_manager = self._get_rois_manager(tuple(self.displayed_image_keys))
#         if roi_manager.is_pre_computed_data_available():
#             self.display_selected_field()
#
#     def status_color_fct(self, status_image_keys):
#         """
#         For a list of image keys, return the color that should be displayed in the combo box.
#         Red: not displayed yet, Green: displayed yet (TODO: Orage: loaded from a file)
#         :param image_keys:
#         :return:
#         """
#         image_keys_to_checked = []
#         for image_keys in self.all_image_keys:
#             # for each status_image_keys we check if it is included in the image_keys
#             status_image_keys_in = True
#             for index, status_image_key in enumerate(status_image_keys):
#                 if status_image_key != image_keys[index]:
#                     status_image_keys_in = False
#                     break
#             if status_image_keys_in:
#                 image_keys_to_checked.append(image_keys)
#
#         if len(image_keys_to_checked) == 0:
#             return "red"
#         at_least_one_not_displayed = False
#         at_least_one_without_saved_data_available = False
#         for image_keys in image_keys_to_checked:
#             roi_manager = self.rois_manager_dict[image_keys]
#             if not roi_manager.are_rois_loaded() and (not roi_manager.is_pre_computed_data_available()):
#                 at_least_one_not_displayed = True
#             if not roi_manager.is_pre_computed_data_available():
#                 at_least_one_without_saved_data_available = True
#         if at_least_one_not_displayed:
#             return "red"
#         if at_least_one_without_saved_data_available:
#             return "green"
#         return "orange"
#
#     def layer_value_changed(self, value):
#         """
#             Called when self.layer_spin_box value is changed
#             Returns:
#
#         """
#         self.current_layer = value
#         for cells_display_widget in self.cells_display_widgets:
#             cells_display_widget.set_layer(self.current_layer)
#         # print(f"layer_value_changed {value}")
#
#     def add_pg_roi(self, pos, image_keys):
#         """
#         Create a new ROI, and either a new cell associated to it if no cell is selected.
#         Args:
#             pos:
#             image_keys:
#
#         Returns:
#
#         """
#         roi_manager = self.rois_manager_dict[tuple(image_keys)]
#         size_half_square = 5
#         # create a new ROI with a square shape
#         contours = list()
#         contours.append([pos[0] - size_half_square, pos[1] + size_half_square])
#         contours.append([pos[0] + size_half_square, pos[1] + size_half_square])
#         contours.append([pos[0] + size_half_square, pos[1] - size_half_square])
#         contours.append([pos[0] - size_half_square, pos[1] - size_half_square])
#
#         # from https://stackoverflow.com/questions/32092899/plot-equation-showing-a-circle/32093458
#         # theta goes from 0 to 2pi
#         n_points = 10
#         theta = np.linspace(0, 2 * np.pi, n_points)
#
#         # the radius of the circle
#         r = np.sqrt(30)
#
#         # compute x1 and x2
#         x_values = r * np.cos(theta) + pos[0]
#         y_values = r * np.sin(theta) + pos[1]
#
#         contours = list()
#         for x_value, y_value in zip(x_values, y_values):
#             contours.append([x_value, y_value])
#
#         roi_manager.add_pg_roi(contours=contours, layer=self.current_layer)
#
#     def change_layer(self, increment=None, layer_index=None):
#         """
#                 increment or decrement layer, or choose directly which layer to display
#                 Args:
#                     increment: bool or None to choose the layer index directly
#                     layer_index: int or None to decrement or increment layer
#
#                 Returns:
#
#         """
#         if increment is not None:
#             if increment:
#                 if self.current_layer + 1 > (self.n_layers - 1):
#                     return
#                 self.current_layer += 1
#                 self.layer_spin_box.setValue(self.current_layer)
#             else:
#                 if self.current_layer - 1 < 0:
#                     return
#                 self.current_layer -= 1
#                 self.layer_spin_box.setValue(self.current_layer)
#         elif layer_index is not None:
#             if layer_index < 0 or layer_index >= self.n_layers:
#                 return
#             self.current_layer = layer_index
#             self.layer_spin_box.setValue(self.current_layer)
#
#     def _get_rois_manager(self, image_keys):
#         # TODO: load roi manage from saved contours
#         roi_manager = self.rois_manager_dict[image_keys]
#         if not roi_manager.are_rois_loaded():
#             # first we check is saved data has been loaded, then we use it
#             if roi_manager.is_pre_computed_data_available():
#                 roi_manager.load_saved_data()
#             else:
#                 # if not yet loaded then we load the rois from the mask data
#                 data_dict = get_data_in_dict_from_keys(list_keys=image_keys, data_dict=self.images_dict)
#                 mask_imgs = get_image_from_tiff(file_name=data_dict["mask"])
#                 roi_manager.load_rois_coordinates_from_masks(mask_imgs=mask_imgs)
#
#         return roi_manager
#
#     def display_selected_field(self):
#         """
#         Display the selected field and change the status of the previously displayed (as seen)
#         :return:
#         """
#
#         self.displayed_image_keys = self.combo_box.get_value()
#         roi_manager = self._get_rois_manager(tuple(self.displayed_image_keys))
#         for cells_display_widget in self.cells_display_widgets:
#             cells_display_widget.set_images(self.displayed_image_keys, roi_manager)
#         # now we update colors
#         roi_manager.update_colors()
#         self.combo_box.update_combo_boxes_status()
#         self.update_cells_buttons_layout()
#
#
# class MyViewBox(pg.ViewBox):
#     """
#     Mixed between RectMode and PanMode.
#     Left click drag act like in RectMode
#     Right click drag act life left click will act in PanMode (move the view box)
#     Allow to zoom.
#     Code from pyqtgraph examples
#     """
#
#     def __init__(self, *args, **kwds):
#         pg.ViewBox.__init__(self, *args, **kwds)
#         # self.setMouseMode(self.RectMode) ViewBox.PanMode
#
#     def mouseClickEvent(self, ev):
#         pass
#         ## reimplement right-click to zoom out
#         # if ev.button() == QtCore.Qt.RightButton:
#         #     self.autoRange()
#
#     def mouseDragEvent(self, ev):
#         """
#         Right click is used to zoom, left click is use to move the area
#         Args:
#             ev:
#
#         Returns:
#
#         """
#         if ev.button() == QtCore.Qt.RightButton:
#             self.setMouseMode(self.PanMode)
#             # cheating, by telling it the left button is used instead
#             ev._buttons = [QtCore.Qt.LeftButton]
#             ev._button = QtCore.Qt.LeftButton
#             pg.ViewBox.mouseDragEvent(self, ev)
#         elif ev.button() == QtCore.Qt.LeftButton:
#             pass
#             # self.setMouseMode(self.RectMode)
#             # pg.ViewBox.mouseDragEvent(self, ev)
#         else:
#             # ev.ignore()
#             pg.ViewBox.mouseDragEvent(self, ev)
#
#
# class ZViewWidget(pg.PlotWidget):
#
#     def __init__(self, n_layers, current_layer, images_dict, main_window, width_image, parent,
#                  linked_cells_display_widget):
#
#         self.view_box = MyViewBox()
#
#         pg.PlotWidget.__init__(self, parent=parent, viewBox=self.view_box)
#
#         # to update x range
#         self.linked_cells_display_widget = linked_cells_display_widget
#         self.current_layer = current_layer
#         self.n_layers = n_layers
#         self.main_window = main_window
#         self.images = None
#         self.images_dict = images_dict
#         # height of the image, used to put y coord proportional
#         self.current_image_height = 100
#         self.cells_color = BREWER_COLORS
#
#         self.roi_manager = None
#         self.image_keys = None
#         # key is the id of the xy ROI,
#         self.line_segments_rois = dict()
#         # key is the id of xy ROI
#         self.layer_rois = dict()
#
#         self.pg_plot = self.getPlotItem()
#
#         # self.pg_plot.hideAxis(axis='left')
#
#         self.pg_plot.hideAxis(axis='bottom')
#
#         self.view_box.setLimits(xMin=0, xMax=width_image, yMin=-1, yMax=self.n_layers)
#         self.pg_plot.setXRange(0, width_image)
#         self.pg_plot.setYRange(0, self.n_layers - 1)
#
#         self.pg_plot.setAspectLocked(True)
#
#         self.layer_color_pen = pg.mkPen(color=(0, 0, 255), width=1)
#         self.current_layer_marker = pg.InfiniteLine(pos=[0, self.current_layer], angle=0,
#                                                     pen=self.layer_color_pen, movable=False)
#         self.pg_plot.addItem(item=self.current_layer_marker)
#
#         self.lines_grid = dict()
#         # white dot line
#         mk_pen = pg.mkPen(color=(255, 255, 255), style=QtCore.Qt.DashLine, width=0.5)
#         for layer in range(self.n_layers - 1):
#             grid_line = pg.InfiniteLine(pos=[0, layer + 0.5], angle=0,
#                                         pen=mk_pen, movable=False)
#             self.pg_plot.addItem(item=grid_line)
#             self.lines_grid[layer] = grid_line
#
#         # test_line = pg.InfiniteLine(pos=[100, 0], angle=90,
#         #                                             pen=color_pen, movable=False)
#         # self.pg_plot.addItem(item=test_line)
#
#     def set_layer(self, layer):
#         self.current_layer = layer
#         self.pg_plot.removeItem(item=self.current_layer_marker)
#
#         self.current_layer_marker = pg.InfiniteLine(pos=[0, self.current_layer], angle=0,
#                                                     pen=self.layer_color_pen, movable=False)
#         self.pg_plot.addItem(item=self.current_layer_marker)
#
#     def change_x_range(self, range_values):
#         self.pg_plot.setXRange(range_values[0], range_values[1])
#
#     # def update_region(self):
#     #     # print("z_stack update_region")
#     #     new_view_range = self.linked_cells_display_widget.view.viewRange()
#     #     actual_view_range = self.pg_plot.view_box.viewRange()
#     #     if (new_view_range[0][0] != actual_view_range[0][0]) or (new_view_range[0][1] != actual_view_range[0][1]):
#     #         self.pg_plot.setXRange(new_view_range[0][0], new_view_range[0][1], padding=0)
#     #     # print(f"z_stack view_range {view_range}")
#
#     def set_images(self, image_keys, roi_manager):
#         """
#         image_keys: List of string
#         """
#
#         data_dict = get_data_in_dict_from_keys(list_keys=image_keys, data_dict=self.images_dict)
#         self.images = get_image_from_tiff(file_name=data_dict["mask"])
#         image = self.images[0]
#         width = image.shape[1]
#         self.current_image_height = image.shape[1]
#         self.change_x_range((0, width))
#
#         self.roi_manager = roi_manager
#         self.image_keys = image_keys
#         self._update_pg_rois()
#
#         # # print(f"key_image {self.last_image_key} {data_dict[self.last_image_key]}")
#         #
#         # self._update_display()
#
#     def _update_pg_rois(self):
#         """
#         Get all pg rois from roi_manager and create LineSegmentROI in each layer to represent them
#         :return:
#         """
#         # pg as pyqtgraph
#         old_line_segments_rois = self.line_segments_rois
#         self.line_segments_rois = dict()
#         self.layer_rois = dict()
#         for layer in np.arange(0, self.n_layers):
#             new_pg_rois = self.roi_manager.get_pg_rois(cells_display_key="red", layer_index=layer)
#             if len(old_line_segments_rois) > 0:
#                 for line_roi in old_line_segments_rois.values():
#                     self.pg_plot.removeItem(line_roi)
#             for pg_roi in new_pg_rois:
#                 self.update_associated_line(pg_roi, layer)
#
#     def update_line_color(self, pg_roi):
#         color = self.cells_color[pg_roi.cell_id % len(self.cells_color)]
#         line_sgt = self.line_segments_rois[pg_roi.roi_id]
#         line_pen = pg.mkPen(color=color, width=4)
#         line_sgt.setPen(line_pen)
#
#     def update_associated_line(self, pg_roi, layer=None):
#         """
#         Update the line associated to this pg_roi. If line doesn't exists, it is created
#         :param pg_roi:
#         :return:
#         """
#         pass
#         # print(f"pg_roi.boundingRect {pg_roi.boundingRect().getCoords()}")
#
#         # getCoords:  position of the rectangle's top-left and bottom-right corner
#         if pg_roi.roi_id in self.line_segments_rois:
#             # if present we remove it, and build a new one, might not be the most efficient way
#             # but the easiest to code
#             line_sgt = self.line_segments_rois[pg_roi.roi_id]
#             layer = self.layer_rois[pg_roi.roi_id]
#             self.pg_plot.removeItem(line_sgt)
#
#         if layer is None:
#             return
#
#         rect_coords = pg_roi.boundingRect().getCoords()
#         # position comparing to creation
#         pos_x = pg_roi.state['pos'][0]
#         pos_y = pg_roi.state['pos'][1]
#         # print(f"update_associated_line roi_id {pg_roi.roi_id}, layer {layer} pos {pg_roi.state['pos']} {rect_coords}")
#         left_x = rect_coords[0] + pos_x
#         right_x = rect_coords[2] + pos_x
#         # calculating y, we center it in the y value corresponding to layer
#         mean_y = (rect_coords[1] + rect_coords[3] + 2 * pos_y) / 2
#         # scale it from 0 to 1
#         y_coord = 1 - (mean_y / self.current_image_height)
#         # 0 is 0.5 under the layer
#         y_coord = layer - 0.5 + y_coord
#
#         color = self.cells_color[pg_roi.cell_id % len(self.cells_color)]
#         line_pen = pg.mkPen(color=color, width=4)
#         line_sgt = pg.LineSegmentROI([[left_x, y_coord], [right_x, y_coord]], pen=line_pen)
#
#         self.pg_plot.addItem(line_sgt)
#         self.line_segments_rois[pg_roi.roi_id] = line_sgt
#         self.layer_rois[pg_roi.roi_id] = layer
#
#     def delete_associated_line(self, roi_id):
#         """
#         Delete the line associated to it
#         :param roi_id:
#         :return:
#         """
#         if roi_id not in self.line_segments_rois:
#             return
#         line_sgt = self.line_segments_rois[roi_id]
#         self.pg_plot.removeItem(line_sgt)
#         del self.line_segments_rois[roi_id]
#         del self.layer_rois[roi_id]
#
#     def link_to_view(self, view_to_link):
#         self.view_box.setXLink(view=view_to_link)


# class CellsDisplayMainWidget(pg.GraphicsLayoutWidget):
#     """
#     Module that will display the different w intervals along the frames
#     """
#
#     def __init__(self, id_widget, current_z, images_dict, key_image, main_window, central_widget, parent=None):
#
#         # self.view_box = MyViewBox()
#         pg.GraphicsLayoutWidget.__init__(self)  # viewBox=self.view_box
#         # allows the widget to be expanded in both axis
#         self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         # self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
#         self.current_layer = current_z
#         # see get_tiff_names for the structure
#         # dict with: group, f, position, s, depth, key_image
#         self.images_dict = images_dict
#         self.id_widget = id_widget
#         # ex: cfos, red, mask
#         self.last_image_key = key_image
#         self.main_window = main_window
#         self.central_widget = central_widget
#         # list of string representing the images actually displayed
#         self.image_keys = None
#
#         self.view = self.addViewBox(lockAspect=True, row=0, col=0, invertY=True, name=f"{id_widget}")
#         # view.setMenuEnabled(False)
#
#         self.image_displayed = pg.ImageItem(axisOrder='row-major', border='w')
#         self.view.addItem(self.image_displayed)
#
#         # different layer
#         self.images = None
#         # list of pyqtgraph rois
#         self.current_pg_rois = []
#         self.roi_manager = None
#
#     def load_contours(self, contours):
#         pass
#
#     def set_images(self, image_keys, roi_manager):
#         """
#         image_keys: List of string
#         """
#         self.roi_manager = roi_manager
#         self.image_keys = image_keys
#         self._update_pg_rois()
#
#         data_dict = get_data_in_dict_from_keys(list_keys=image_keys, data_dict=self.images_dict)
#         self.images = get_image_from_tiff(file_name=data_dict[self.id_widget])
#         # print(f"key_image {self.last_image_key} {data_dict[self.last_image_key]}")
#
#         self._update_display()
#
#     def remove_pg_roi(self, pg_roi):
#         self._remove_item_from_view_safe(pg_roi)
#
#     def add_pg_roi(self, pg_roi):
#         """
#         Should be called from RoiManager, pg_roi should be registered in the RoiManager
#         Args:
#             pg_roi:
#
#         Returns:
#
#         """
#         self.view.addItem(pg_roi)
#
#     def _update_pg_rois(self):
#         # pg as pyqtgraph
#         new_pg_rois = self.roi_manager.get_pg_rois(cells_display_key=self.id_widget, layer_index=self.current_layer)
#         if len(self.current_pg_rois) > 0:
#             for pg_roi in self.current_pg_rois:
#                 self._remove_item_from_view_safe(pg_roi)
#         self.current_pg_rois = new_pg_rois
#         for pg_roi in self.current_pg_rois:
#             self.view.addItem(pg_roi)
#
#     def _remove_item_from_view_safe(self, pg_roi):
#         # modificaiton of the source code of self.view.removeItem(pg_roi)
#         # to avoid this;
#         # QGraphicsScene::removeItem: item 0x7f82f6625f80's scene (0x0) is different from this scene (0x7f82f63418e0)
#         exception_raised = False
#         try:
#             self.view.addedItems.remove(pg_roi)
#         except:
#             exception_raised = True
#         if not exception_raised:
#             self.view.scene().removeItem(pg_roi)
#             self.view.updateAutoRange()
#
#     def set_layer(self, layer):
#         self.current_layer = layer
#         self._update_pg_rois()
#
#         self._update_display()
#
#     def _update_display(self):
#         image_to_display = self.images[self.current_layer]
#         if self.last_image_key == "mask":
#             # otherwise the mask with no cell will be white instead of black
#             if len(np.where(image_to_display == 0)[0]) > 4:
#                 image_to_display = np.invert(image_to_display)
#             # print(
#             #     f"image_to_display sum {np.sum(image_to_display)} {image_to_display.shape[0] * image_to_display.shape[1]} "
#             #     f"min {np.min(image_to_display)}, max {np.max(image_to_display)}, "
#             #     f"len(np.where(image_to_display == np.min(image_to_display)[0]) "
#             #     f"{len(np.where(image_to_display == np.min(image_to_display))[0])}, "
#             #     f"len(np.where(image_to_display == np.max(image_to_display)[0]) "
#             #     f"{len(np.where(image_to_display == np.max(image_to_display))[0])}")
#
#         #     image_to_display = np.reshape(image_to_display, (image_to_display.shape[0], image_to_display.shape[1], 1))
#         # print(f"id_widget {self.id_widget}, image shape {image_to_display.shape}")
#         self.image_displayed.setImage(image_to_display)
#
#     def create_new_roi_in_the_middle(self):
#         view_range = self.view.viewRange()
#         # putting the new ROI in the middle
#         pos = [np.mean(view_range[0]), np.mean(view_range[1])]
#         self.central_widget.add_pg_roi(pos=pos, image_keys=self.image_keys)
#
#     def keyPressEvent(self, event):
#         """
#         Call when a key is pressed
#         Args:
#             event:
#
#         Returns:
#
#         """
#         if event.key() == QtCore.Qt.Key_N:
#             self.create_new_roi_in_the_middle()
#         if (event.modifiers() & Qt.ControlModifier) and (event.key() == QtCore.Qt.Key_V):
#             self.roi_manager.paste_roi(layer=self.current_layer)
#         if (event.modifiers() & Qt.ControlModifier) and (event.key() == QtCore.Qt.Key_C):
#             self.roi_manager.copy_hover_roi()
#         if (event.modifiers() & Qt.ControlModifier) and (event.key() == QtCore.Qt.Key_R):
#             self.roi_manager.remove_hover_roi()
#         if event.key() == QtCore.Qt.Key_Plus:
#             self.roi_manager.dilate_hover_roi()
#         if event.key() == QtCore.Qt.Key_Minus:
#             self.roi_manager.erode_hover_roi()
#         # Sending the event to the main window if the widget is in the main window
#         if self.main_window is not None:
#             self.main_window.keyPressEvent(event=event)
#
#     def link_to_view(self, view):
#         self.view.setXLink(view=view)
#         self.view.setYLink(view=view)


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
        mask_with_contours = np.reshape(mask_with_contours,
                                        (mask_with_contours.shape[0], mask_with_contours.shape[1], 1))
    # contours, hierarchy = cv2.findContours(mask_with_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    """
    https://stackoverflow.com/questions/39475125/compatibility-issue-with-contourarea-in-opencv-3
    in OpenCV 3.4.X, cv2.findContours() returns 3 items

    image, contours, hierarchy = cv2.findContours()
    In OpenCV 2.X and 4.1.X, cv2.findContours() returns 2 items

    """
    contours, hierarchy = cv2.findContours(mask_with_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # CHAIN_APPROX_SIMPLE CHAIN_APPROX_TC89_L1

    coord_contours = []
    centroids = []
    for contour_index, contour in enumerate(contours):
        # if contour_index < 2:
        #     print(f"contour {contour}")
        area = cv2.contourArea(contour)
        # perimeter = cv2.arcLength(contour, True)
        # Contour approximation
        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        epsilon = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        xy = []
        for c in contour:
            xy.append([c[0][0] + 0.5, c[0][1] + 0.5])
        # removing the contour that take all the frame
        if [0.5, 0.5] in xy:
            continue
        if area < 5:
            # removing one pixel cells
            continue

        coord_contours.append(xy)
        # centroid
        m = cv2.moments(contour)
        cx = int(m['m10'] / m['m00'])
        cy = int(m['m01'] / m['m00'])
        centroids.append((cx, cy))

    return coord_contours, centroids


def convert_contours_to_cv2_format(contours):
    """
    Take a list of tuple of 2 values (x, y) and transform it in a shape compatible with cv2
    :param contours:
    :return:
    """
    cv2_contours = []

    for xy in contours:
        cv2_contours.append([[int(xy[0]), int(xy[1])]])
    print(f'cv2_contours {np.array(cv2_contours)}')
    return np.array(cv2_contours)


class PlanMask:

    def __init__(self, mask_data, mask_id, result_path):
        self.mask_data = mask_data
        self.mask_id = mask_id
        self.result_path = result_path
        # plt.imshow(mask_data)
        # plt.show()
        mask_with_contours = mask_data.copy()
        mask_with_contours = np.reshape(mask_with_contours,
                                        (mask_with_contours.shape[0], mask_with_contours.shape[1], 1))
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



def save_results_in_xls_file(result_path, data_dict):
    """

    Args:
        result_path:
        data_dict: the key is a tuple representing the image, value is a dict with key is an int representing the cell
         than value is a dict with each key the field description and value the corresponding value

    Returns:

    """
    image_keys_names = ["group", "letter", "dors_ventr", "s", "position"]

    # just to get column names
    fields_names = []
    for image_keys, cell_dict in data_dict.items():
        if image_keys == ("summary", ):
            continue
        for fields_dict in cell_dict.values():
            fields_names.extend(list(fields_dict.keys()))
            break
        break
    fields_names.extend(list(data_dict[("summary", )].keys()))


    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    writer = pd.ExcelWriter(f'{result_path}/rosie_data_{time_str}.xlsx')
    column_names = []
    column_names.extend(image_keys_names)
    column_names.append("cell")
    column_names.extend(fields_names)

    results_df = pd.DataFrame(columns=column_names)
    line_index = 1
    for image_keys, cell_dict in data_dict.items():
        if image_keys == ("summary", ):
            results_df.at[line_index, "group"] = "Summary"
            for key_summary, value_summary in cell_dict.items():
                results_df.at[line_index, key_summary] = value_summary
            line_index += 1
            continue
        if len(cell_dict) == 0:
            continue

        for cell, fields_dict in cell_dict.items():
            for image_key_index, image_key in enumerate(image_keys):
                results_df.at[line_index, image_keys_names[image_key_index]] = image_key
            results_df.at[line_index, "cell"] = cell
            for field_key, field_value in fields_dict.items():
                results_df.at[line_index, field_key] = field_value
            line_index += 1

    results_df.to_excel(writer, 'data', index=False)
    writer.save()


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

root_path = 'C:/Users/cavalieri/Documents/INMED/Current Experiments/CFOS/'
mask_dir_path = os.path.join(root_path, "Detection", "masques")
red_dir_path = os.path.join(root_path, "Detection", "cellules (red)")
cfos_dir_path = os.path.join(root_path, "Detection", "cfos_processed (green)")
# cfos_filt_dir_path = os.path.join(root_path, "Detection", "cfos_filtered (green)")

result_path = os.path.join(root_path, "ANALYSIS", "results")

# pickle_file_name = os.path.join(root_path, "pkl_files", "2Dorsal-22-01_lexi.pkl")
pickle_file_name = os.path.join(result_path, "FINAL2.pkl")

with open(pickle_file_name, 'rb') as f:
    loaded_data_dict = pickle.load(f)

# n_layers = 7
# the key is a tuple representing the image
# value is a dict with key is an int representing the cell
# than value is a dict with each key the field description and value the corresponding value
data_dict = SortedDict()
images_dict = get_tiff_names(red_dir_path=red_dir_path, cfos_dir_path=cfos_dir_path,
                             mask_dir_path=mask_dir_path)

all_image_keys = get_tree_dict_as_a_list(images_dict)
# removing the two last keys which are like "mask", "red" and the tiffs file_name
all_image_keys = set([tuple(images[:-2]) for images in all_image_keys])
novel_95p_c_fos_list = []

# Add lists for the groups
group_a_novel_99p_c_fos_list = []
group_b_novel_99p_c_fos_list = []
group_c_novel_99p_c_fos_list = []
group_d_novel_99p_c_fos_list = []
group_e_novel_99p_c_fos_list = []

print(f"N images {len(loaded_data_dict)}")




for image_keys, pre_computed_data in loaded_data_dict.items():
    images_data_dict = get_data_in_dict_from_keys(list_keys=image_keys, data_dict=images_dict)
    cfos_images = get_image_from_tiff(file_name=images_data_dict["cfos"])
    data_dict[image_keys] = SortedDict()
    print(f"{image_keys}:")
    # novel condition
    # if image_keys[1].lower().strip().startswith("n"):
    #     novel_95p_c_fos_list.append(np.percentile(cfos_images, 95))
    #     if image_keys[0] == 'GroupA':
    #         group_a_novel_99p_c_fos_list.append(np.percentile(cfos_images, 99))
    #     elif image_keys[0] == 'GroupB':
    #         group_b_novel_99p_c_fos_list.append(np.percentile(cfos_images, 99))
    #     elif image_keys[0] == 'GroupC':
    #         group_c_novel_99p_c_fos_list.append(np.percentile(cfos_images, 99))
    #     elif image_keys[0] == 'GroupD':
    #         group_d_novel_99p_c_fos_list.append(np.percentile(cfos_images, 99))
    #     elif image_keys[0] == 'GroupE':
    #         group_e_novel_99p_c_fos_list.append(np.percentile(cfos_images, 99))

    for cell_id, layer_dict in pre_computed_data.items():
        print(f"cell_id {cell_id}")
        data_dict[image_keys][cell_id] = dict()
        cell_dict = data_dict[image_keys][cell_id]
        sum_areas = 0
        sum_pixels_intensity = 0
        sum_pixels_intensity_z_score = 0
        sum_median_pixels_intensity = 0
        sum_median_pixels_intensity_z_score = 0
        # cfos_matrix = np.zeros((len(layer_dict), cfos_images[0].shape[0], cfos_images[0].shape[1]))
        # cfos_matrix_z_score = np.zeros((len(layer_dict), cfos_images[0].shape[0], cfos_images[0].shape[1]))

        for layer, all_contours in enumerate(layer_dict.values()):
            cfos_image = cfos_images[layer]
            # normalizing cfos image, z_score
            cfos_image_original = cfos_image.copy()
            cfos_image_z_score = (cfos_image - np.mean(cfos_image)) / np.std(cfos_image)
            # cfos_matrix[layer] = cfos_image
            # cfos_matrix_z_score[layer] = cfos_image_z_score
            for contours in all_contours:
                # building pixel mask from the contours
                # converting contours as array and value as integers

                contours_array = np.zeros((2, len(contours)), dtype="int16")

                # for contour_index, coord in enumerate(contours):
                #     contours_array[0, contour_index] = int(coord[0])
                #     contours_array[1, contour_index] = int(coord[1])

                mask_image = np.zeros(cfos_image.shape[:2], dtype="bool")
                # morphology.binary_fill_holes(input
                mask_image[contours_array[1, :], contours_array[0, :]] = True

                # area = np.sum(mask_image)

                img = Image.new("L", [512, 512], 0)
                ImageDraw.Draw(img).polygon(contours, outline=1, fill=1)
                mask_fill = np.array(img)

                area=np.sum(mask_fill)
                sum_areas += area

                # pixels_intensity = np.sum(cfos_image_z_score[mask_image])
                # sum_pixels_intensity_z_score += pixels_intensity

                pixels_intensity = np.sum(cfos_image*mask_fill)
                sum_pixels_intensity += pixels_intensity

                sum_median_pixels_intensity += np.median(cfos_image_original[mask_image])

                sum_median_pixels_intensity_z_score += np.median(cfos_image_z_score[mask_image])

        cell_dict["sum_areas"] = sum_areas
        cell_dict["n_layers"] = len(layer_dict)
        cell_dict["mean_cfos_image"] = np.mean(cfos_images)
        cell_dict["median_cfos_image"] = np.median(cfos_images)
        cfos_imges_z_score = (cfos_images - np.mean(cfos_images))
        cfos_imges_z_score = cfos_imges_z_score / np.std(cfos_imges_z_score)
        cell_dict["mean_cfos_image_z_score"] = np.mean(cfos_imges_z_score)

        # cell_dict["95_p_cfos_image"] = np.mean([np.percentile(img, 95) for img in cfos_images])
        cell_dict["95_p_cfos_image"] = np.percentile(cfos_images, 95)
        cell_dict["99_p_cfos_image"] = np.percentile(cfos_images, 99)
        cell_dict["sum_pixels_intensity_z_score"] = sum_pixels_intensity_z_score
        cell_dict["sum_pixels_intensity"] = sum_pixels_intensity
        cell_dict["sum_median_pixels_intensity"] = sum_pixels_intensity
        cell_dict["sum_median_pixels_intensity"] = sum_median_pixels_intensity
        cell_dict["sum_median_pixels_intensity_z_score"] = sum_median_pixels_intensity_z_score
        # cell_dict["legend for summaries"] = ['A', 'B', 'C', 'D', 'E']
summaries = [np.mean(group_a_novel_99p_c_fos_list),
             np.mean(group_b_novel_99p_c_fos_list),
             np.mean(group_c_novel_99p_c_fos_list),
             np.mean(group_d_novel_99p_c_fos_list),
             np.mean(group_e_novel_99p_c_fos_list),
             ]
data_dict[("summary",)] = {"95_p_cfos_novel_image": (str(np.mean(novel_95p_c_fos_list)) + '\n' + str(summaries))}