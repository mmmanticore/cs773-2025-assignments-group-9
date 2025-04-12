from operator import truediv

from PIL import Image
from scipy.sparse.linalg import MatrixRankWarning

import imageProcessing.convolve2D as IPConv2D
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

def average_lvbo(width, height, pixels):
    out = IPUtils.createInitializedGreyscalePixelArray(width,height)
    # If the skeleton code skips edges, change to range(1, image_height-1)
    for r in range(1, height - 1):
        for c in range(1,width - 1):
            s = 0.0
            for rr in range(-1, 2):
                for cc in range(-1, 2):
                    s += pixels[r + rr][c + cc]
                    out[r][c] = s / 9.0
    return out

def load_Grey(image_path):
    img = Image.open(image_path).convert('L')
    width, height = img.size
    pixels = list(img.getdata())

    pixel_array = []
    for y in range(height):
        row = []
        for x in range(width):
            grey = pixels[y * width + x]
            row.append(grey)
        pixel_array.append(row)
    return width, height, pixel_array

#When k_size = 5
def jisuan_Gaosilvbo(width, height, pixels):
    k = [0.0625, 0.25, 0.375, 0.25, 0.0625]  # sigma≈1.0
    averaged = IPConv2D.computeSeparableConvolution2DOddNTapBorderZero(
    pixels, width, height, k
     )
    return averaged

def sobel_core(width, height, pixels):
    sobel_x = [[-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]]
    sobel_y = [[-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]]
    Ix = IPUtils.createInitializedGreyscalePixelArray(width, height)
    Iy = IPUtils.createInitializedGreyscalePixelArray(width, height)
    # 略去边界
    for r in range(1, height - 1):
        for c in range(1,width - 1):
            gx = 0.0
            gy = 0.0
            for rr in range(-1, 2):
                for cc in range(-1, 2):
                    pixel_val = pixels[r + rr][c + cc]
                    gx += pixel_val * sobel_x[rr + 1][cc + 1]
                    gy += pixel_val * sobel_y[rr + 1][cc + 1]
            Ix[r][c] = gx
            Iy[r][c] = gy
    return Ix, Iy

# 2. Harris 角点检测（不使用第三方库）
def computeHarrisCorners(pixels, width, height, k=0.04):
    # 1) 计算梯度 Ix, Iy
    Ix, Iy = sobel_core(width,height,pixels)

    # 2) 初始化 Ixx, Iyy, Ixy
    Ixx = IPUtils.createInitializedGreyscalePixelArray(width, height)
    Iyy = IPUtils.createInitializedGreyscalePixelArray(width, height)
    Ixy = IPUtils.createInitializedGreyscalePixelArray(width,  height)

    R= IPUtils.createInitializedGreyscalePixelArray(width, height)

    for y in range(height):
        for x in range(width):
            xx = Ix[y][x]
            yy = Iy[y][x]
            Ixx[y][x] = xx * xx
            Iyy[y][x] = yy * yy
            Ixy[y][x] = xx * yy

    # 3) 对 Ixx, Iyy, Ixy 进行高斯平滑，减少噪声
    Ixx = jisuan_Gaosilvbo( width, height, Ixx)
    Iyy = jisuan_Gaosilvbo( width, height, Iyy)
    Ixy = jisuan_Gaosilvbo( width, height, Ixy)

    # 4) 计算 Harris 响应 R = det(M) - k*(trace(M))^2
    for y in range(height):
        for x in range(width):
            xx=Ixx[y][x]
            yy=Iyy[y][x]
            xy=Ixy[y][x]
            detM = xx*yy-xy*xy
            traceM = xx+yy
            R[y][x] = detM - k * (traceM ** 2)

    # 5) 提取所有大于阈值的点（或者直接提取全部），记录响应值和坐标
    goodcorners_list = []

    for y in range(height):
        for x in range(width):
            if R[y][x] > 1e7:  # 按题目要求阈值10^7
                goodcorners_list.append(((x, y),R[y][x]))

    print(f"Total corners above threshold: {len(goodcorners_list)}")
    return R,goodcorners_list


def NMS(height, width, R_Matrix,goodcorners_list):
    afterNMS=[]
    for (x,y),R in goodcorners_list:
        is_max = True
        for i in range(-1, 2):
            for j in range(-1, 2):
                n_y=y+i
                n_x=x+j
                if 0 <= n_y < height and 0 <= n_x < width:
                    if R_Matrix[n_y][n_x] > R:
                        is_max = False
                        break
            if not is_max:
                break
        if is_max:
            afterNMS.append(((x, y),R))
    return afterNMS

def sort_by_R(NMScorners, num=1000):
    sorted_corners = sorted(NMScorners, key=lambda item: item[1], reverse=True)

    top_1000_corners = []
    for i, corner in enumerate(sorted_corners):
        if i >= 1000:
            break
        top_1000_corners.append(corner)

    return top_1000_corners


def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"
    width_left, height_left, pixels_left = load_Grey(filename_left_image)
    width_right, height_right, pixels_right = load_Grey(filename_right_image)

    # 均值滤波
    after_av_left=average_lvbo(width_left,height_left,pixels_left)
    after_av_right=average_lvbo(width_right,height_right,pixels_right)

    # make sure greyscale image is stretched to full 8 bit intensity range of 0 to 255
    px_left = IPPixelOps.scaleTo0And255AndQuantize(after_av_left, width_left, height_left)
    px_right = IPPixelOps.scaleTo0And255AndQuantize(after_av_right, width_right, height_right)

    # Harris 检测
    R_left,Hcorners_left = computeHarrisCorners(px_left, width_left, height_left)
    R_right,Hcorners_right = computeHarrisCorners(px_right, width_right, height_right)

    # NMS非极大值抑制
    NMScorners_left = NMS(height_left, width_left,R_left, Hcorners_left)
    NMScorners_right = NMS(height_right, width_right,R_right, Hcorners_right)

    # 根据R值排序找到最大1000个点
    sortcorners_left = sort_by_R(NMScorners_left)
    sortcorners_right = sort_by_R(NMScorners_right)

    fig, axs = pyplot.subplots(1, 2, figsize=(10, 5))
    # Left image + corner point
    axs[0].imshow(px_left, cmap='gray')
    axs[0].set_title("Left with corners")
    for ((x, y),R) in sortcorners_left:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs[0].add_patch(circ)
     # Right picture + corner point
    axs[1].imshow(px_right, cmap='gray')
    axs[1].set_title("Right with corners")
    for ((x, y),R) in sortcorners_right:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs[1].add_patch(circ)

    pyplot.show()

    return sortcorners_left,sortcorners_right,height_left,height_right,width_left,width_right,px_left,px_right


if __name__ == "__main__":
    main()
