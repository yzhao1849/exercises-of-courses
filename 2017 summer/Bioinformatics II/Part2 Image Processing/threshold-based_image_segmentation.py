"""Threshold-Based Image Segmentation"""

import numpy as np
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt

# (a) Read the grayscale image brain.png
brain=misc.imread('brain.png')
fig1=plt.figure()
ax=fig1.add_subplot(2,2,1)
plt.axis('off')
plt.imshow(brain,cmap='gray')
ax=fig1.add_subplot(2,2,2)
plt.axis('off')
ax.set_title("size=2")
brain_denoised2=ndimage.median_filter(brain,size=2)
plt.axis('off')
plt.imshow(brain_denoised2,cmap='gray')
ax=fig1.add_subplot(2,2,3)
plt.axis('off')
ax.set_title("size=4")
brain_denoised=ndimage.median_filter(brain,size=4)
plt.imshow(brain_denoised,cmap='gray')
ax=fig1.add_subplot(2,2,4)
plt.axis('off')
ax.set_title("size=6")
brain_denoised6=ndimage.median_filter(brain,size=6)
plt.imshow(brain_denoised6,cmap='gray')

# (b) Otsu thresholding, and create 3 masks for background, gray matter, and white matter respectively
from skimage import filters
val1 = filters.threshold_otsu(brain_denoised)
val2=filters.threshold_otsu(brain_denoised[brain_denoised>val1])

print("Thresholds:",val1,val2)

mask_bg=brain_denoised<val1
mask_gm=(brain_denoised>val1) & (brain_denoised<val2)
mask_wm=brain_denoised>val2
misc.imsave("brain-bg.png",mask_bg)
misc.imsave("brain-gm.png",mask_gm)
misc.imsave("brain-wm.png",mask_wm)

fig2=plt.figure(figsize=(10, 3))
ax=fig2.add_subplot(1,3,1)
plt.axis('off')
ax.set_title("background")
plt.imshow(mask_bg,cmap='gray')

ax=fig2.add_subplot(1,3,2)
plt.axis('off')
ax.set_title("gray matter")
plt.imshow(mask_gm,cmap='gray')

ax=fig2.add_subplot(1,3,3)
plt.axis('off')
ax.set_title("white matter")
plt.imshow(mask_wm,cmap='gray')

# (c) Plot a log-scaled histogram of the image
plt.figure()
plt.hist(brain_denoised.ravel(),bins=256,range=(0,256),log=True,fc='k',ec='k')
plt.xlim(0,257)
plt.axvline(val1, color='k', ls='--')
plt.axvline(val2, color='k', ls='--')

# (d) Combine the three masks into a single color image
color_brain=np.zeros(brain_denoised.shape,dtype=np.uint8)
color_brain[mask_gm]=255
color_brain[mask_bg]=256//2
plt.figure()
plt.axis('off')
plt.imshow(color_brain,cmap='brg')

# (e) Use erosion filter to produce a border between the gray and white matter
import copy
with_border=copy.deepcopy(brain_denoised)
rev_mask_wm=brain_denoised<=val2
mask_border=ndimage.binary_erosion(mask_wm,structure=np.ones((2,2))).astype(bool) \
            == ndimage.binary_erosion(rev_mask_wm,structure=np.ones((2,2))).astype(bool) #both will be False on the border
with_border[mask_border]=0
plt.figure()
plt.axis('off')
plt.imshow(with_border,cmap='gray')



# (f) Use bilinear interpolation to upsample the image by a factor of 4 along each axis
def plot_upsample(method,size):
    brain_resized = misc.imresize(brain_denoised, size=size, interp=method)
    resized_bg = brain_resized < val1
    resized_gm = (brain_resized > val1) & (brain_resized < val2)
    resized_wm = brain_resized > val2
    mask_bg_resized = misc.imresize(mask_bg, size=size, interp=method)
    mask_gm_resized = misc.imresize(mask_gm, size=size, interp=method)
    mask_wm_resized = misc.imresize(mask_wm, size=size, interp=method)
    fig = plt.figure(figsize=(6, 9))
    fig.suptitle(method)
    ax = fig.add_subplot(3, 2, 1)
    plt.axis('off')
    ax.set_title("resized_bg")
    plt.imshow(resized_bg, cmap='gray')
    ax = fig.add_subplot(3, 2, 2)
    plt.axis('off')
    ax.set_title("mask_bg_resized")
    plt.imshow(mask_bg_resized, cmap='gray')
    ax = fig.add_subplot(3, 2, 3)
    plt.axis('off')
    ax.set_title("resized_gm")
    plt.imshow(resized_gm, cmap='gray')
    ax = fig.add_subplot(3, 2, 4)
    plt.axis('off')
    ax.set_title("mask_gm_resized")
    plt.imshow(mask_gm_resized, cmap='gray')
    ax = fig.add_subplot(3, 2, 5)
    plt.axis('off')
    ax.set_title("resized_wm")
    plt.imshow(resized_wm, cmap='gray')
    ax = fig.add_subplot(3, 2, 6)
    plt.axis('off')
    ax.set_title("mask_wm_resized")
    plt.imshow(mask_wm_resized, cmap='gray')

plot_upsample('bilinear',4.0)
plot_upsample('nearest',4.0)

plt.show()
