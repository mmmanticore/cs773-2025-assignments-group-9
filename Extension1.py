import math

import random
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
from timeit import default_timer as timer
from scipy.ndimage import distance_transform_edt as edt, binary_dilation
import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth


# SUSAN Corners
import numpy as np


def computeSUSANCorners(px_array, image_width, image_height, radius=3, threshold=20, geometric_threshold=0.7,
                        max_corners=1000):

    #  Calculate corner point response
    response_map = np.zeros((image_height, image_width))
    offsets = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]

    for y in range(radius, image_height - radius):
        for x in range(radius, image_width - radius):
            center_intensity = px_array[y][x]
            similar_count = sum(
                1 for dy, dx in offsets
                if abs(px_array[y + dy][x + dx] - center_intensity) < threshold
            )

            # Calculate SUSAN corner point response
            response_map[y, x] = 1 - (similar_count / len(offsets))

    #Corner filtering (only corners above the threshold are selected)
    max_response = np.max(response_map)
    corner_threshold = geometric_threshold * max_response
    corners = [(x, y) for y in range(image_height) for x in range(image_width) if response_map[y, x] > corner_threshold]

    print(f"Number of corners after preliminary screening: {len(corners)}")

    # NMS: 3x3
    nms_corners = []
    corner_responses = []
    window_size = 5
    offset = window_size // 2

    for (cx, cy) in corners:
        r, c = cy, cx
        current_val = response_map[r, c]
        is_max = True
        for rr in range(-offset, offset + 1):
            for cc in range(-offset, offset + 1):
                nr = r + rr
                nc = c + cc
                if nr < 0 or nr >= image_height or nc < 0 or nc >= image_width:
                    continue
                if response_map[nr, nc] > current_val:
                    is_max = False
                    break
            if not is_max:
                break
        if is_max:
            nms_corners.append((cx, cy))
            corner_responses.append(current_val)

    print(f"Number of corners after NMS: {len(nms_corners)}")

    # Sorting and extracting the top 1000 strongest corners
    if len(nms_corners) > max_corners:
        top_indices = np.argsort(corner_responses)[::-1][:max_corners]  # 按角点响应降序排序
        final_corners = [nms_corners[i] for i in top_indices]
    else:
        final_corners = nms_corners

    print(f"final corners number: {len(final_corners)}")

    return nms_corners


def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    (image_width, image_height, px_array_left_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_left_image)
    (image_width, image_height, px_array_right_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_right_image)

    # 计算 SUSAN 角点
    start = timer()
    SUSAN_corners_left = computeSUSANCorners(px_array_left_original, image_width, image_height)
    SUSAN_corners_right = computeSUSANCorners(px_array_right_original, image_width, image_height)
    end = timer()
    print(f"SUSAN calculate time: {end - start:.4f} s")
    print(f"detect {len(SUSAN_corners_left)} corners")
    print(f"detect {len(SUSAN_corners_right)} corners")

    # save as csv file
    def save_corners_to_csv(corners, output_path):
        with open(output_path, 'w') as f:
            f.write("Index,X,Y\n")
            for idx, (x, y) in enumerate(corners):
                f.write(f"{idx},{x},{y}\n")
        print(f"save as {output_path}")

    save_corners_to_csv(SUSAN_corners_left, "SUSAN_corners_left.csv")
    save_corners_to_csv(SUSAN_corners_right, "SUSAN_corners_right.csv")

    # draw corners on images using matplotlib
    fig1, axs1 = pyplot.subplots(1, 2, figsize=(10, 5))
    axs1[0].imshow(px_array_left_original, cmap='gray')
    axs1[0].set_title("Left image with Harris corners")
    for (x, y) in SUSAN_corners_left:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs1[0].add_patch(circ)
    axs1[1].imshow(px_array_right_original, cmap='gray')
    axs1[1].set_title("Right image with Harris corners")
    for (x, y) in SUSAN_corners_right:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs1[1].add_patch(circ)

    pyplot.show()



if __name__ == "__main__":
    main()