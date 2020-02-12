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
import pickle
from shapely.geometry import Polygon, MultiPoint
import pandas as pd
from datetime import datetime

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

    def save_input_in_pickle(pickle_file, data_dict):
        with open(pickle_file, 'wb') as f:
            pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
        save_results_in_xls_file(result_path, data_dict)

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

def get_data_in_dict_from_keys(list_keys, data_dict):
    if len(list_keys) > 0:
        return get_data_in_dict_from_keys(list_keys[1:], data_dict[list_keys[0]])
    return data_dict

def save_input_in_pickle(pickle_file, inputs_dict):
    with open(pickle_file, 'wb') as f:
        pickle.dump(inputs_dict, f, pickle.HIGHEST_PROTOCOL)
    save_results_in_xls_file2(result_path, inputs_dict)

def save_results_in_xls_file2(result_path, inputs_dict):
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
    for image_keys, cell_dict in inputs_dict.items():
        if image_keys == ("summary", ):
            continue
        for fields_dict in cell_dict.values():
                    if hasattr(fields_dict, 'keys'):
                        fields_names.extend(list(inputs_dict.keys()))
        # if hasattr(inputs_dict[("summary", )], 'keys'):
        # fields_names.extend(list(inputs_dict.keys()))


    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")

    writer = pd.ExcelWriter(f'{result_path}/DB_data_{time_str}.xlsx')
    column_names = []
    column_names.extend(image_keys_names)
    column_names.append("cell")
    column_names.extend(fields_names)
    column_names.append("DOUBLE STAINING")

    results_df = pd.DataFrame(columns=column_names)
    line_index = 1
    for image_keys, cell_dict in inputs_dict.items():
        if bool(cell_dict)==True:
            if image_keys == ("summary", ):
                results_df.at[line_index, "group"] = "Summary"
                for key_summary, value_summary in cell_dict.items():
                    results_df.at[line_index, key_summary] = value_summary
                line_index += 1
                continue
            if len(cell_dict) == 0:
                continue

            for cell, fields_dict in cell_dict.items():
                if hasattr(cell_dict,'keys'):
                    for image_key_index, image_key in enumerate(image_keys):
                        results_df.at[line_index, image_keys_names[image_key_index]] = image_key
                    results_df.at[line_index, "cell"] = cell
                    results_df.at[line_index, "DOUBLE STAINING"] = fields_dict
                        # for field_key, field_value in enumerate(fields_dict):
                        #     results_df.at[line_index, field_key] = field_value
                    line_index += 1
                else:
                    line_index += 1
        results_df.to_excel(writer, 'data', index=False)
        writer.save()

def plot_manual_data(pickle_file_name, mask_dir_path, red_dir_path, cfos_dir_path, result_path,
                     input_pickle_file_name):
    """
    Analyse the data that has been saved using the GUI
    :return:
    """

    # def press_fig_img(event):
    #     print(f"press {event.key}")


    with open(pickle_file_name, 'rb') as f:
        loaded_data_dict = pickle.load(f)

    if os.path.isfile(input_pickle_file_name):
        with open(input_pickle_file_name, 'rb') as f:
            inputs_dict = pickle.load(f)
    else:
        inputs_dict = dict()

    try:
        data_dict = SortedDict()
        images_dict = get_tiff_names(red_dir_path=red_dir_path, cfos_dir_path=cfos_dir_path,
                                     mask_dir_path=mask_dir_path)

        all_image_keys = get_tree_dict_as_a_list(images_dict)
        # removing the two last keys which are like "mask", "red" and the tiffs file_name
        all_image_keys = set([tuple(images[:-2]) for images in all_image_keys])

        # save_input_in_pickle(input_pickle_file_name, inputs_dict)

        for image_keys, pre_computed_data in loaded_data_dict.items():
            if image_keys in inputs_dict:
                continue
            else:
                inputs_dict[image_keys] = dict()
            images_data_dict = get_data_in_dict_from_keys(list_keys=image_keys, data_dict=images_dict)
            cfos_images = get_image_from_tiff(file_name=images_data_dict["cfos"])
            red_images = get_image_from_tiff(file_name=images_data_dict["red"])
            data_dict[image_keys] = SortedDict()
            print(f"{image_keys}:")

            for cell_id, layer_dict in pre_computed_data.items():
                data_dict[image_keys][cell_id] = dict()
                cell_dict = data_dict[image_keys][cell_id]

                layer_sum = len(layer_dict)
                print(f"layer_dict {list(layer_dict.keys())}")
                cfos_layers = np.empty((layer_sum, cfos_images.shape[1], cfos_images.shape[2]))
                cfos_layers[:] = np.nan
                K=0
                for layer, all_contours in layer_dict.items():
                    try:
                        cfos_layers[K, :, :] = cfos_images[layer]
                    except Exception as exep:
                        print(exep)
                    # print(f"cfos_layers {cfos_layers}")
                    K=K+1
                cfos_image = cfos_layers.sum(axis=0)

                middle_cont = int(np.round(np.mean(list(layer_dict.keys()))))

                print(f"middle contour: {middle_cont}")
                contours = layer_dict[middle_cont]

                cell_polygon = patches.Polygon(xy=contours[0],
                                               fill=False, linewidth=1,
                                               facecolor=None,
                                               edgecolor="yellow",
                                               zorder=10)  # lw=2

                # for contours in all_contours:
                #     # building pixel mask from the contours
                #     # converting contours as array and value as integers
                #     contours_array = np.zeros((2, len(contours)), dtype="int16")
                #     for contour_index, coord in enumerate(contours):
                #         contours_array[0, contour_index] = int(coord[0])
                #         contours_array[1, contour_index] = int(coord[1])
                #     mask_image = np.zeros(cfos_image.shape[:2], dtype="bool")
                #     # morphology.binary_fill_holes(input
                #     mask_image[contours_array[1, :], contours_array[0, :]] = True
                #     print(f"Slice: {image_keys} Cell ID:{cell_id} Z-level:{layer}")
                #     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
                #     fig.canvas.set_window_title(f"Slice: {image_keys} Cell ID:{cell_id} Z-level:{layer}")
                #     ax1.imshow(mask_image)

                # m = cv2.moments(contours)
                # # centroid
                # cx = int(m['m10'] / m['m00'])
                # cy = int(m['m01'] / m['m00'])
                poly_gon = MultiPoint(contours[0])
                cx, cy = poly_gon.centroid.coords[0]

                padding = 25
                # [max(0, cx-padding): min(cfos_image.shape[0], cx+padding+1),
                #                            max(0, cy-padding): min(cfos_image.shape[1], cy+padding+1)]
                fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)
                fig.canvas.set_window_title(f"{image_keys} Cell ID:{cell_id} layers:{layer_sum}")
                # fig.canvas.mpl_connect("key_press_event", press_fig_img)
                ax1.imshow(cfos_image)
                ax1.add_patch(cell_polygon)
                ax2.imshow(red_images[middle_cont])
                ax1.set_xlim(max(0, cx - padding), min(cfos_image.shape[0], cx + padding + 1))
                ax1.set_ylim(max(0, cy - padding), min(cfos_image.shape[1], cy + padding + 1))

                plt.savefig(os.path.join(result_path, "images", f"{image_keys}_{cell_id}.png"))
                plt.show()
                plt.close()

                input_str = input("Tell me everything: ")
                while input_str not in ["y", "n", "m", "end"]:
                    input_str = input("Tell me everything: ")
                if input_str == "end":
                    save_input_in_pickle(input_pickle_file_name, inputs_dict)
                    return
                inputs_dict[image_keys][cell_id] = input_str
    except Exception as exep:
        print("Exception catched " + str(exep))
        save_input_in_pickle(input_pickle_file_name, inputs_dict)

    save_input_in_pickle(input_pickle_file_name, inputs_dict)


root_path='D:/INMED/CFOS/lexi/'
# result_path = 'C:/Users/cavalieri/Documents/INMED/Current Experiments/CFOS/ANALYSIS/results'
# input_pickle_file_name=os.path.join(result_path,'double_staining.pkl')

mask_dir_path = os.path.join(root_path, "masques")
red_dir_path = os.path.join(root_path, "cellules (red)")
cfos_dir_path = os.path.join(root_path, "cfos (green)")

result_path = os.path.join(root_path, "results_ld")

# pickle_file_name = os.path.join(root_path, "pkl_files", "2Dorsal-22-01_lexi.pkl")
pickle_file_name = os.path.join(result_path, "FIN-FIN.pkl")
input_pickle_file_name = os.path.join(result_path, "double_staining.pkl")

# main_gui(mask_dir_path, red_dir_path, cfos_dir_path)
# analyse_manual_data(pickle_file_name, mask_dir_path, red_dir_path, cfos_dir_path, result_path)
plot_manual_data(pickle_file_name, mask_dir_path, red_dir_path, cfos_dir_path, result_path, input_pickle_file_name)
# input_pickle_file_name=os.path.join(result_path, "double_staining.pkl"))


# for i,j in list(loaded_data_dict.items()):
# ...    print(f'i {i}')
# ...    for k in list(j):
# ...        print(f'k {k}')