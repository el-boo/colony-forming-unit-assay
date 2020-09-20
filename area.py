import sys
import glob
import numpy as np
import skimage.io
import skimage.color
import matplotlib
import matplotlib.pyplot as plt
from skimage import exposure, util
from skimage.filters import threshold_otsu


def crop_center_fraction(img, fraction):
    y, x = img.shape
    off = int((min(img.shape) * fraction) // 2)
    start_x = x // 2 - off
    start_y = y // 2 - off
    return img[start_y:start_y + 2 * off, start_x:start_x + 2 * off]


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def main(file_name):
    img = util.invert(skimage.io.imread(file_name, as_gray=True))
    # rectangular crop
    img = crop_center_fraction(img, 0.9)
    h, w = img.shape[:2]
    # zero everything outside the center circle
    r_roi = h * 0.45
    mask = create_circular_mask(h, w, radius=r_roi)
    img = exposure.equalize_adapthist(img)

    # img = img > threshold_otsu(img)
    img = img > 0.4
    img[~mask] = 0
    area = 100 * np.count_nonzero(img) / (np.pi * (r_roi ** 2))

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    ax.imshow(img)
    fig.suptitle(f'area: {area:.2f} %')
    ax.set_axis_off()

    plt.savefig(file_name + '_area.png', bbox_inches='tight', dpi=1000)


if __name__ == '__main__':
    matplotlib.rc('text', usetex=False)
    d = sys.argv[1]
    for filename in glob.iglob(d + '**/**', recursive=True):
        if filename.endswith('.TIF'):
            print(f'processing {filename}')
            main(filename)
