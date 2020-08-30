import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
from skimage import exposure
from skimage.feature import blob_doh
from skimage.filters import sobel


def crop_center_fraction(img, fraction):
    y, x = img.shape
    off = int((min(img.shape) * fraction) // 2)
    start_x = x // 2 - off
    start_y = y // 2 - off
    return img[start_y:start_y + 2*off, start_x:start_x + 2*off]


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
    matplotlib.rc('text', usetex=False)

    img = skimage.io.imread(file_name, as_gray=True)
    # rectangular crop
    img = crop_center_fraction(img, 0.9)
    h, w = img.shape[:2]
    # zero everything outside the center circle
    r_roi = h * 0.45
    mask = create_circular_mask(h, w, radius=r_roi)
    img[~mask] = 0

    # preprocessing - increase contrast
    img = exposure.equalize_adapthist(img)
    img = sobel(img)
    img = exposure.equalize_adapthist(img)

    # find blobs
    blobs_doh = blob_doh(img, min_sigma=15, max_sigma=40, num_sigma=40, threshold=0.0004, log_scale=False, overlap=.5)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.imshow(img)
    ax.set_axis_off()

    area = 0
    count = 0
    for blob in blobs_doh:
        y, x, r = map(int, blob)
        a = r // 2
        # ignore bad ones - less intense than the image on average
        p = img[y - a:y + a, x - a:x + a].ravel()
        if np.mean(p) < img.mean():
            continue
        ax.add_patch(plt.Circle((x, y), r, color='red', linewidth=0.4, fill=False))
        area += r ** 2
        count += 1
    fig.suptitle(f'count: {count}, area: {int(100 * area // r_roi**2)} %')
    plt.savefig(file_name+'_processed.png', bbox_inches='tight', dpi=1000)
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
