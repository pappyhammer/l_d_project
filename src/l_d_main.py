import numpy as np
import os
import PIL
from ScanImageTiffReader import ScanImageTiffReader
from PIL import ImageSequence
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches


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
            # ax.add_patch(cell_polygon)

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

if __name__ == "__main__":
    main()
