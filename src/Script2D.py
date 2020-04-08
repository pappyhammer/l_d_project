import time
start = time.time()
print("CODE RUNNING")

import mxnet as mx
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from cellpose import utils, models,io,plot
import skimage.io

dir="C:/Users/cavalieri/Desktop/CFOS/JULIEN/DAPI"
os.chdir(dir)

#PARAMETERS
mod="cyto"
chan=[0,0]
diam=10.
threshold=0.4

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

#execute cellpose
model = models.Cellpose(device, model_type=mod) #model_type='cyto' or model_type='nuclei'
masks, flows, styles, diams = model.eval(imgs, diameter=diam, channels=chan, threshold=threshold)
utils.masks_flows_to_seg(imgs, masks, flows, diams, chan, imlist)

#io.save_png(masks)
# utils.save_to_png(masks, masks, flows, imlist)

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