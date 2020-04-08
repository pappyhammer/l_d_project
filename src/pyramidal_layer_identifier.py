import PIL
from ScanImageTiffReader import ScanImageTiffReader
from PIL import ImageSequence
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import tifffile
import hdbscan
from bisect import bisect_right
import seaborn as sns
from shapely import geometry
from matplotlib import patches
import math
import networkx as nx



# qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12 + 11 diverting
BREWER_COLORS = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                 '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                 '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                 '#74add1', '#4575b4', '#313695']

def plot_hist_distribution(distribution_data, description, param=None, values_to_scatter=None,
                           xticks_labelsize=10, yticks_labelsize=10, x_label_font_size=15, y_label_font_size=15,
                           labels=None, scatter_shapes=None, colors=None, tight_x_range=False,
                           twice_more_bins=False, background_color="black", labels_color="white",
                           xlabel="", ylabel=None, path_results=None, save_formats="pdf",
                           v_line=None, x_range=None,
                           ax_to_use=None, color_to_use=None):
    """
    Plot a distribution in the form of an histogram, with option for adding some scatter values
    :param distribution_data:
    :param description:
    :param param:
    :param values_to_scatter:
    :param labels:
    :param scatter_shapes:
    :param colors:
    :param tight_x_range:
    :param twice_more_bins:
    :param xlabel:
    :param ylabel:
    :param save_formats:
    :return:
    """
    distribution = np.array(distribution_data)
    if color_to_use is None:
        hist_color = "blue"
    else:
        hist_color = color_to_use
    edge_color = "white"
    if x_range is not None:
        min_range = x_range[0]
        max_range = x_range[1]
    elif tight_x_range:
        max_range = np.max(distribution)
        min_range = np.min(distribution)
    else:
        max_range = 100
        min_range = 0
    weights = (np.ones_like(distribution) / (len(distribution))) * 100
    if ax_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        ax1.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
    else:
        ax1 = ax_to_use
    bins = int(np.sqrt(len(distribution)))
    if twice_more_bins:
        bins *= 2
    hist_plt, edges_plt, patches_plt = ax1.hist(distribution, bins=bins, range=(min_range, max_range),
                                                facecolor=hist_color,
                                                edgecolor=edge_color,
                                                weights=weights, log=False, label=description)
    if values_to_scatter is not None:
        scatter_bins = np.ones(len(values_to_scatter), dtype="int16")
        scatter_bins *= -1

        for i, edge in enumerate(edges_plt):
            # print(f"i {i}, edge {edge}")
            if i >= len(hist_plt):
                # means that scatter left are on the edge of the last bin
                scatter_bins[scatter_bins == -1] = i - 1
                break

            if len(values_to_scatter[values_to_scatter <= edge]) > 0:
                if (i + 1) < len(edges_plt):
                    bool_list = values_to_scatter < edge  # edges_plt[i + 1]
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                new_i = max(0, i - 1)
                                scatter_bins[i_bool] = new_i
                else:
                    bool_list = values_to_scatter < edge
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                scatter_bins[i_bool] = i

        decay = np.linspace(1.1, 1.15, len(values_to_scatter))
        for i, value_to_scatter in enumerate(values_to_scatter):
            if i < len(labels):
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20, label=labels[i])
            else:
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20)
    y_min, y_max = ax1.get_ylim()
    if v_line is not None:
        ax1.vlines(v_line, y_min, y_max,
                   color="white", linewidth=2,
                   linestyles="dashed", zorder=5)

    ax1.legend()

    if tight_x_range:
        ax1.set_xlim(min_range, max_range)
    else:
        ax1.set_xlim(0, 100)
        xticks = np.arange(0, 110, 10)

        ax1.set_xticks(xticks)
        # sce clusters labels
        ax1.set_xticklabels(xticks)
    ax1.yaxis.set_tick_params(labelsize=xticks_labelsize)
    ax1.xaxis.set_tick_params(labelsize=yticks_labelsize)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    # TO remove the ticks but not the labels
    # ax1.xaxis.set_ticks_position('none')

    if ylabel is None:
        ax1.set_ylabel("Distribution (%)", fontsize=30, labelpad=20)
    else:
        ax1.set_ylabel(ylabel, fontsize=y_label_font_size, labelpad=20)
    ax1.set_xlabel(xlabel, fontsize=x_label_font_size, labelpad=20)

    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)

    # padding between ticks label and  label axis
    # ax1.tick_params(axis='both', which='major', pad=15)

    if ax_to_use is None:
        fig.tight_layout()
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        if path_results is None:
            path_results = param.path_results
        time_str = ""
        if param is not None:
            time_str = param.time_str
        for save_format in save_formats:
            fig.savefig(f'{path_results}/{description}'
                        f'_{time_str}.{save_format}',
                        format=f"{save_format}",
                                facecolor=fig.get_facecolor())

        plt.close()



def save_array_as_tiff(array_to_save, path_results, file_name):
    """

    :param array_to_save:
    :param path_results:
    :param file_name:
    :return:
    """
    # then saving each frame as a unique tiff
    tiff_file_name = os.path.join(path_results, file_name)
    with tifffile.TiffWriter(tiff_file_name) as tiff:
        tiff.save(array_to_save, compress=0)


def binarized_frame(movie_frame, filled_value=1, percentile_threshold=90, threshold_value=None, with_uint=True):
    """
    Take a 2d-array and return a binarized version, thresholding using a percentile value.
    It could be filled with 1 or another value
    Args:
        movie_frame:
        filled_value:
        percentile_threshold:

    Returns:

    """
    img = np.copy(movie_frame)
    if threshold_value is None:
        threshold = np.percentile(img, percentile_threshold)
    else:
        threshold = threshold_value

    img[img < threshold] = 0
    img[img >= threshold] = filled_value

    if with_uint:
        img = img.astype("uint8")
    else:
        img = img.astype("int8")
    return img


def load_tiff_movie(tiff_file_name):
    """
    Load a tiff movie from tiff file name.
    Args:
        tiff_file_name:

    Returns: a 3d array: n_frames * width_FOV * height_FOV

    """
    try:
        # start_time = time.time()
        tiff_movie = ScanImageTiffReader(tiff_file_name).data()
        # stop_time = time.time()
        # print(f"Time for loading movie with ScanImageTiffReader: "
        #       f"{np.round(stop_time - start_time, 3)} s")
    except Exception as e:
        im = PIL.Image.open(tiff_file_name)
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_y, dim_x = np.array(im).shape
        tiff_movie = np.zeros((n_frames, dim_y, dim_x), dtype="uint16")
        for frame, page in enumerate(ImageSequence.Iterator(im)):
            tiff_movie[frame] = np.array(page)
    return tiff_movie


def find_blobs(tiff_array, results_path=None, result_id=None, with_blob_display=False):
    # img = ((tiff_array - tiff_array.min()) * (1 / (tiff_array.max() - tiff_array.min()) * 255)).astype('uint8')
    img = binarized_frame(movie_frame=tiff_array, filled_value=255, percentile_threshold=85)
    # img = tiff_array.copy()
    # # plt.imshow(img)
    # # plt.show()
    # threshold = np.percentile(img, 98)
    #
    # img[img < threshold] = 0
    # img[img >= threshold] = 255
    # # print(tiff_array.shape)
    #
    # # print(f"im min {np.min(tiff_array[0])} max {np.max(tiff_array[0])}")
    # img = img.astype("uint8")

    # img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    # print(img.shape)
    # Detect blobs.

    # invert image (blob detection only works with white background)
    img = cv2.bitwise_not(img)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 10
    # params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 25

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.3

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # plt.imshow(tiff_array[0], cmap=cm.Reds)
    # plt.show()
    # Set up the detector with default parameters.
    ## check opencv version and construct the detector
    is_v2 = cv2.__version__.startswith("2.")
    if is_v2:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(img)
    # print(f"len keypoints {len(keypoints)}")

    # Draw detected blobs as red circles.
    if with_blob_display:
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show keypoints
        # cv2.imshow("Keypoints", im_with_keypoints)
        cv2.imwrite(os.path.join(results_path, f"{result_id}_blobs.png"), im_with_keypoints)
        cv2.waitKey(0)

    from shapely.geometry import Polygon

    centers_of_gravity = np.zeros((len(keypoints), 2))
    for index_keypoint, keypoint in enumerate(keypoints):
        centers_of_gravity[index_keypoint, 0] = keypoint.pt[1]
        centers_of_gravity[index_keypoint, 1] = keypoint.pt[0]

    # centers_of_gravity = [(keypoint.pt[0], keypoint.pt[1]) for keypoint in keypoints]
    # for keypoint in keypoints:
    #     x = keypoint.pt[0]
    #     y = keypoint.pt[1]
    #     s = keypoint.size
    #
    #     # polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    #     # polygon.centroid
    #     print(f"x {x}, y {y}")
    return centers_of_gravity

def analyze_movie(frame_img, results_id, results_path):
    # print(f"tiff_array.mean {np.mean(tiff_array)}")

    new_results_path = os.path.join(results_path, results_id[:-2])
    if not os.path.exists(new_results_path):
        os.mkdir(new_results_path)

    # each element if a tuple of 2 int represent the center of a cell
    # TODO: See to find blob on avg movie ? np.mean(movie, axis=0)
    # centers_of_gravity = find_blobs(movie.copy()[0])
    centers_of_gravity = find_blobs(frame_img, with_blob_display=True,
                                    results_path=new_results_path, result_id=results_id)

    print(f"Nb of blobs: {len(centers_of_gravity)}")
    print(f"centers_of_gravity.shape {centers_of_gravity.shape}")

    # plt.scatter(*centers_of_gravity.T, s=50, linewidth=0, c='b', alpha=0.25)
    # plt.show()
    cluster_cells(cells_coords=centers_of_gravity.T, results_id=results_id,
                  results_path=new_results_path)

    # plt.show()
    # raise Exception("TOTO")
    # cells_movie = []


def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = np.arctan(delta[1] / delta[0])
    if delta[0] < 0:
        res += np.pi
    return res


def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    return np.linalg.norm(np.cross((p2 - p1), (p3 - p1))) / 2.

def convex_hull(points, smidgen=0.0075):
    '''
    from: https://stackoverflow.com/questions/17553035/draw-a-smooth-polygon-around-data-points-in-a-scatter-plot-in-matplotlib
    Calculate subset of points that make a convex hull around points
    Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

    :Parameters:
    points : ndarray (2 x m)
    array of points for which to find hull
    use pylab to show progress?
    smidgen : float
    offset for graphic number labels - useful values depend on your data range

    :Returns:
    hull_points : ndarray (2 x n)
    convex hull surrounding points
    '''

    n_pts = points.shape[1]
    # assert(n_pts > 5)
    centre = points.mean(1)

    angles = np.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:, angles.argsort()]

    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i], pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i], pts[(i + 2) % n_pts])
            if Aij + Ajk < Aik:
                del pts[i + 1]
            i += 1
            n_pts = len(pts)
        k += 1
    return np.asarray(pts)

def cluster_cells_with_hdbscan(cells_coords, results_id, results_path, img_dim=None):
    plot_scatter_with_clusters = True

    print("")
    print(f"cluster_cells() for {results_id}")
    print(f"N cells {len(cells_coords)}")
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                cluster_selection_epsilon=0.0,
                                metric='euclidean', min_cluster_size=20, min_samples=20, p=None)
    # metric='precomputed' euclidean
    clusterer.fit(cells_coords)

    clusters_unique_labels = np.unique(clusterer.labels_)
    # print(f"clusters_unique_labels {clusters_unique_labels}")
    n_clusters = len(clusters_unique_labels)
    if len(np.where(clusterer.labels_ == -1)[0]) > 0:
        n_clusters -= 1

    # print(f"{results_id} n_clusters {n_clusters}")
    # plt.imshow(img) #, cmap=cm.Greys)
    # plt.show()
    scatter_size = 20
    if n_clusters == 0:
        if plot_scatter_with_clusters:
            # color_palette = sns.color_palette('deep', 8)
            cluster_colors = [BREWER_COLORS[x] if x >= 0
                              else (0, 0, 0)  # (0.5, 0.5, 0.5) for gray
                              for x in clusterer.labels_]
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                     zip(cluster_colors, clusterer.probabilities_)]
            fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                    gridspec_kw={'height_ratios': [1]},
                                    figsize=(12, 12))
            ax1.imshow(img)
            ax1.scatter(*cells_coords.T, s=scatter_size, linewidth=0, c=cluster_member_colors, alpha=0.95)
            # plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
            fig.savefig(f'{results_path}/hdbscan_cluster_{results_id}.png',
                        format=f"png",
                        facecolor=fig.get_facecolor())

            plt.close()
        return

    biggest_cluster_label = None
    biggest_cluters_size = 0
    for cluster_label in clusters_unique_labels:
        if cluster_label == -1:
            continue
        n_elems = len(np.where(clusterer.labels_ == cluster_label)[0])
        if biggest_cluters_size < n_elems:
            biggest_cluters_size = n_elems
            biggest_cluster_label = cluster_label

    print(f"/// biggest_cluters_size {biggest_cluters_size}")
    cells_in_cluster = np.where(clusterer.labels_ == biggest_cluster_label)[0]

    if img_dim is None:
        return

    # masks = np.zeros(img_dim, dtype="int8")
    # for cell_in_cluster in cells_in_cluster:
    #     coord = cells_coords[cell_in_cluster]
    #     masks[coord[0], coord[1]] = 1
    #
    # outlines = plot.outlines_list(segmentation_data['masks'])
    # outline_pyr_layer = outlines[0]
    # print(f"outline_pyr_layer {outline_pyr_layer}")

    points = np.zeros((2, len(cells_in_cluster)))
    for cell_index, cell_in_cluster in enumerate(cells_in_cluster):
        c_x, c_y = cells_coords[cell_in_cluster]
        points[0, cell_index] = c_x
        points[1, cell_index] = c_y
    # finding the convex_hull for each group
    xy = convex_hull(points=points)
    # xy = xy.transpose()
    # print(f"xy {xy}")
    # xy is a numpy array with as many line as polygon point
    # and 2 columns: x and y coords of each point

    if plot_scatter_with_clusters:
        # color_palette = sns.color_palette('deep', 8)
        cluster_colors = ["red" if x == biggest_cluster_label else BREWER_COLORS[x] if x >= 0
        else (0, 0, 0)  # (0.5, 0.5, 0.5) for gray
                          for x in clusterer.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, clusterer.probabilities_)]
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        ax1.imshow(img)
        ax1.scatter(*cells_coords.T, s=scatter_size, linewidth=0, c=cluster_member_colors, alpha=0.95)
        # ax1.plot(outline_pyr_layer[:, 0], outline_pyr_layer[:, 1], color='r')
        poly_gon = patches.Polygon(xy=xy,
                                   fill=False, linewidth=0,
                                   edgecolor="red",
                                   zorder=15, lw=3)
        ax1.add_patch(poly_gon)
        # plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        fig.savefig(f'{results_path}/hdbscan_cluster_{results_id}.png',
                    format=f"png",
                    facecolor=fig.get_facecolor())

        plt.close()

def build_distance_adjacency_matrix(cells_coords):
    """

        Args:
            cells_coords: 2 array (n_cells, 2) represents x, y center of gravity coord

        Returns:

        """

    n_cells = len(cells_coords)

    adjacency_matrix = np.zeros((n_cells, n_cells))

    for cell_1 in np.arange(n_cells-1):
        for cell_2 in np.arange(cell_1, n_cells):
            x_1, y_1 = cells_coords[cell_1][0], cells_coords[cell_1][1]
            x_2, y_2 = cells_coords[cell_2][0], cells_coords[cell_2][1]

            distance = math.sqrt(((x_1 - x_2) ** 2) + ((y_1 - y_2) ** 2))
            adjacency_matrix[cell_1, cell_2] = distance
    return adjacency_matrix

def cluster_cells_with_graph(cells_coords, results_id, results_path, img_dim=None):
    n_cells = len(cells_coords)
    adjacency_matrix = build_distance_adjacency_matrix(cells_coords=cells_coords)
    # now we need to put a threshold in order to keep in the graph only edges
    # of cells that are nearby
    # low_dist = np.percentile(adjacency_matrix[adjacency_matrix >0], 1)
    # print(f"low_dist {low_dist}")
    dist_threshold = 22.5
    adjacency_matrix[adjacency_matrix >= dist_threshold] = 0
    # 25 as a low threhsold ?
    weight_on_edges = False
    graph = nx.Graph()
    graph.add_nodes_from(np.arange(n_cells))
    for cell in np.arange(n_cells):
        conected_cells = np.where(adjacency_matrix[cell] > 0)[0]
        if weight_on_edges:
            # with weight
            edges = [(cell, c_cell, adjacency_matrix[cell, c_cell]) for c_cell in conected_cells]
            graph.add_weighted_edges_from(edges)
        else:
            edges = [(cell, c_cell) for c_cell in conected_cells]
            graph.add_edges_from(edges)

    # TODO: build a recursive fct that remove all edges from node with only one edge, run until no change
    #  is done anymore

    plot_graph = True
    node_size = 20
    if plot_graph:
        # color_palette = sns.color_palette('deep', 8)
        cluster_colors = ["red"] * n_cells
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        ax1.imshow(img)
        positions = {cell: (cells_coords[cell, 0], cells_coords[cell, 1]) for cell in range(n_cells)}
        node_color = "white"
        edge_color = "red"
        nx.draw_networkx(graph, pos=positions, node_size=node_size, edge_color=edge_color,
                         cmap=None,
                         node_color=node_color, arrowsize=4, width=1,
                         with_labels=False, arrows=False,
                         ax=ax1)
        # ax1.scatter(*cells_coords.T, s=scatter_size, linewidth=0, c=cluster_colors, alpha=0.95)
        # plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        fig.savefig(f'{results_path}/graph_{results_id}.png',
                    format=f"png",
                    facecolor=fig.get_facecolor())

        plt.close()

def cluster_cells(cells_coords, results_id, results_path, img_dim=None, using_hdbscan=False):
    """

    Args:
        cells_coords: 2 array (n_cells, 2) represents x, y center of gravity coord
        results_id:
        results_path:
        img_dim: (int, int)

    Returns:

    """

    if using_hdbscan:
        cluster_cells_with_hdbscan(cells_coords=cells_coords, results_id=results_id,
                                   results_path=results_path, img_dim=img_dim)
    else:
        cluster_cells_with_graph(cells_coords=cells_coords, results_id=results_id,
                                   results_path=results_path, img_dim=img_dim)



def build_cell_polygon_from_contour(cell_coord):
    """
    Build the (shapely) polygon representing a given cell using its contour's coordinates.
    Args:
        cell_coord: 2d array (2, n_coord)

    Returns:

    """

    # make a list of tuple representing x, y coordinates of the contours points
    coord_list_tuple = list(zip(cell_coord[0], cell_coord[1]))

    # buffer(0) or convex_hull could be used if the coords are a list of points not
    # in the right order. However buffer(0) return a MultiPolygon with no coords available.
    if len(coord_list_tuple) < 3:
        list_points = []
        for coords in coord_list_tuple:
            list_points.append(geometry.Point(coords))
        return geometry.LineString(list_points)
    else:
        return geometry.Polygon(coord_list_tuple)

if __name__ == '__main__':
    root_path = "/media/julien/Not_today/davide_lexi_project/davide_pyramidal/"

    use_cellpose_segmentation = True
    if use_cellpose_segmentation:
        data_path = os.path.join(root_path, "data/cellpose_segmentation/cellpose_results/")
    else:
        data_path = os.path.join(root_path, "data/DAPI/")

    results_path = os.path.join(root_path, "results")
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    results_path = os.path.join(results_path, time_str)
    os.mkdir(results_path)

    file_names = []
    # look for filenames in the fisrst directory, if we don't break, it will go through all directories
    for (dirpath, dirnames, local_filenames) in os.walk(data_path):
        file_names.extend(local_filenames)
        break

    if use_cellpose_segmentation:
        # keeping only npy files
        file_names = [f for f in file_names if (not f.startswith(".")) and f.endswith(".npy")]
        for file_name in file_names:
            identifier = file_name[:-4]
            segmentation_data = np.load(os.path.join(data_path, file_name), allow_pickle=True).item()
            outlines = segmentation_data["outlines"]

            from cellpose import plot
            img = segmentation_data['img']
            # print(f"img {img.shape}")
            outlines = plot.outlines_list(segmentation_data['masks'])
            centroids = np.zeros((2, len(outlines)), dtype="int16")
            for cell_index, outline in enumerate(outlines):
                polygon = build_cell_polygon_from_contour(cell_coord=outline.T)
                centroid = polygon.centroid.coords[0]
                # print(f"centroid {centroid}")
                centroids[0, cell_index] = centroid[0]
                centroids[1, cell_index] = centroid[1]

            cluster_cells(cells_coords=centroids.T, results_id=identifier, results_path=results_path,
                          img_dim=img.shape)
    else:
        # keeping only tif files
        file_names = [f for f in file_names if (not f.startswith(".")) and f.endswith(".tif")]

        for file_name in file_names:
            identifier = file_name[:-4]
            movie = load_tiff_movie(tiff_file_name=os.path.join(data_path, file_name))
            print(movie.shape)
            for index_layer, frame_img in enumerate(movie):
                analyze_movie(frame_img=frame_img,
                              results_id=identifier + f"_{index_layer}", results_path=results_path)

