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
import A1P1



def jisuan_SUSANCorners(pixel, height, width, window_size=7, threshold=27, area_threshold=0.7,max_corners=1000):
    #  Calculate corner point response
    R_map = np.zeros((height, width))
    half_window_size=window_size // 2

    for y in range(height):
        for x in range(width):
            if x - half_window_size < 0 or x + half_window_size >= width:
                continue
            if y - half_window_size < 0 or y + half_window_size >= height:
                continue

            center_pixel = pixel[y][x]
            similar_count =  0

            for i in range(-half_window_size,half_window_size+1):
                for j in range(-half_window_size,half_window_size+1):
                    ny=y+i
                    nx=x+j
                    if abs(pixel[ny][nx]-center_pixel) < threshold:
                        similar_count +=1

            all_pixels = window_size*window_size
            R=1-(similar_count/all_pixels)
            R_map[y][x]=R

    max_R=np.max(R_map)
    Susan_corners=[]

    for y in range(height):
        for x in range(width):
            if R_map[y][x] > max_R * area_threshold:
                Susan_corners.append(((x,y),R_map[y][x]))

    print(f"Number of corners after preliminary screening: {len(Susan_corners)}")

    return R_map,Susan_corners




def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    width_left, height_left, pixels_left = A1P1.load_Grey(filename_left_image)
    width_right, height_right, pixels_right = A1P1.load_Grey(filename_right_image)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_left = IPPixelOps.scaleTo0And255AndQuantize(pixels_left, width_left, height_left)
    px_right = IPPixelOps.scaleTo0And255AndQuantize(pixels_right, width_right, height_right)

    # 计算 SUSAN 角点
    start = timer()
    R_left,SUSAN_corners_left = jisuan_SUSANCorners(px_left, height_left, width_left)
    R_right,SUSAN_corners_right = jisuan_SUSANCorners(px_right, height_right, width_right)
    end = timer()
    print(f"SUSAN calculate time: {end - start:.4f} s")
    print(f"detect {len(SUSAN_corners_left)} corners")
    print(f"detect {len(SUSAN_corners_right)} corners")

    NMS_SUSAN_corners_left = A1P1.NMS(height_left,width_left,R_left,SUSAN_corners_left)
    NMS_SUSAN_corners_right = A1P1.NMS(height_right, width_right,R_right, SUSAN_corners_right)

    sort_SUSANcorners_left = A1P1.sort_by_R(NMS_SUSAN_corners_left)
    sort_SUSANcorners_right = A1P1.sort_by_R(NMS_SUSAN_corners_right)

    # draw corners on images using matplotlib
    fig, axs = pyplot.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(px_left, cmap='gray')
    axs[0].set_title("Left image with Harris corners")
    for ((x, y),R) in sort_SUSANcorners_left:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs[0].add_patch(circ)
    axs[1].imshow(px_right, cmap='gray')
    axs[1].set_title("Right image with Harris corners")
    for ((x, y),R) in sort_SUSANcorners_right:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs[1].add_patch(circ)

    pyplot.show()
    return sort_SUSANcorners_left, sort_SUSANcorners_right, height_left, height_right, width_left, width_right, px_left, px_right


if __name__ == "__main__":
    main()