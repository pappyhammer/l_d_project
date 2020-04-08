import time
start = time.time()
print("CODE RUNNING")

import mxnet as mx
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from cellpose import plot, transforms,utils, models, io
import skimage

dir="C:/Users/cavalieri/Desktop/CFOS/JULIEN/TdTomato/TEST3D"
os.chdir(dir)

#PARAMETERS
mod="cyto"
chan=[0,0]
diam=18.
split=False   #split 3D image in several stacks (or not)

## GPU CHECK
use_gpu = utils.use_gpu()
if use_gpu:
    device = mx.gpu()
else:
    device = mx.cpu()
print('>>>> using %s'%(['CPU', 'GPU'][use_gpu]))

#images
imlist= glob.glob('*.tif')
imgs = [skimage.io.imread(f) for f in imlist]
nimg = len(imgs)

# check if 3D
shapes = []
for ind, img in enumerate(imgs):
    if len(imgs[ind].shape) & min(imgs[ind].shape) > 1:
        shapes.append(True)
    else:
        shapes.append(False)

if any(shapes) == False:
    exit('Some or all images are NOT stacks')

if split == True: #split 3D image in several stacks (or not)

    #list of images, each being a list of 2D stacks
    list_of_images = []
    for i,im in enumerate(imgs):
        sublist = []
        for j,im in enumerate(imgs[i]):
            sublist.append(imgs[i][j])
        list_of_images.append(sublist)

    #execute cellpose
    model = models.Cellpose(device, model_type=mod) #model_type='cyto' or model_type='nuclei'
    for i, im in enumerate(list_of_images):
        imlist3D = [os.path.splitext(imlist[i])[0] + '_z' + str(n) for n in range(1, len(im) + 1)]
        masks, flows, styles, diams = model.eval(im, diameter=diam, channels=chan)
        utils.masks_flows_to_seg(im, masks, flows, diams, chan, imlist3D)

        imlist3Dtif = [imlist3D[n] + '.tif' for n in range(len(imlist3D))] #saving single-Z plan pictures
        for ind, imag in enumerate(im):
            skimage.io.imsave(imlist3Dtif[ind],imag)

else:
    # execute cellpose
    model = models.Cellpose(device, model_type=mod)  # model_type='cyto' or model_type='nuclei'
    masks, flows, styles, diams = model.eval(imgs, diameter=diam, channels=chan, do_3D=True)
    io.masks_flows_to_seg(imgs, masks, flows, diams, chan, imlist)


# #plot images and save
# for idx,file_name in enumerate(imlist):
#     print(idx)
#     img = transforms.reshape(imgs[idx], chan)
#     img = plot.rgb_image(img)
#     maski = masks[idx]
#     flowi = flows[idx][0]
#
#     fig = plt.figure(figsize=(12,3))
#     plot.show_segmentation(fig, img, maski, flowi)
#     plt.tight_layout()
#     plt.show()
#
#     save_dir = os.path.join(dir, str(idx+1)+"_fig")
#     mask_dir = os.path.join(dir, str(idx + 1))
#     fig.savefig(save_dir, dpi=300) #save combined image
#     skimage.io.imsave(mask_dir + '_cp_masks.png', maski.astype(np.uint16)) #save masks
#     plt.close(fig)


end = time.time()
print("Elaspsed time: " + str(round(end - start)) + " seconds")