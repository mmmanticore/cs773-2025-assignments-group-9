from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
from timeit import default_timer as timer

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth



# 1. 计算图像在 x 和 y 方向上的梯度（Sobel）
def computePartialDerivativesSobel(px_array, image_width, image_height):
    """
    使用 3x3 Sobel 算子(Sobel 算子是一种常用的边缘检测算子，用于提取图像的边缘信息。它通过计算图像中每个像素点的梯度，来检测图像中灰度值变化最大的区域，
    也就是图像的边缘。Sobel 算子基于图像的梯度，即在水平方向和垂直方向上计算图像强度的变化。Sobel 算子有两个主要的方向：水平边缘检测和垂直边缘检测。
    每个方向的 Sobel 算子都是一个 3x3 的卷积核。)来计算 Ix, Iy.
    px_array: 2D list, 灰度图像像素.
    返回 (Ix, Iy)，它们也是 2D list.
    """
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    sobel_y = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]
    Ix = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    Iy = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    # 略去边界
    for r in range(1, image_height - 1):
        for c in range(1, image_width - 1):
            gx = 0.0
            gy = 0.0
            for rr in range(-1, 2):
                for cc in range(-1, 2):
                    pixel_val = px_array[r + rr][c + cc]
                    gx += pixel_val * sobel_x[rr + 1][cc + 1]
                    gy += pixel_val * sobel_y[rr + 1][cc + 1]
            Ix[r][c] = gx
            Iy[r][c] = gy
    return Ix, Iy



# 2. Harris 角点检测（不使用第三方库）
def computeHarrisCorners(px_array, image_width, image_height, k=0.05, threshold_ratio=0.01, do_nonmax_suppression=True):
    """
    计算 Harris 角点响应，并返回 (x, y) 格式的角点坐标列表。
    参数：
      - k: Harris 公式中的参数，一般取 0.04~0.06。
      - threshold_ratio: 阈值比例（相对于最大响应），例如 0.01 表示 1%。
      - do_nonmax_suppression: 是否进行简单的非极大值抑制。
    """
    # 1) 计算梯度 Ix, Iy
    Ix, Iy = computePartialDerivativesSobel(px_array, image_width, image_height)

    # 2) 计算 Ixx, Iyy, Ixy
    Ixx = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    Iyy = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    Ixy = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(image_height):
        for c in range(image_width):
            Ixx[r][c] = Ix[r][c] * Ix[r][c]
            Iyy[r][c] = Iy[r][c] * Iy[r][c]
            Ixy[r][c] = Ix[r][c] * Iy[r][c]

    # 3) 对 Ixx, Iyy, Ixy 进行高斯平滑，减少噪声
    Ixx = IPSmooth.computeGaussianAveraging3x3(Ixx, image_width, image_height)
    Iyy = IPSmooth.computeGaussianAveraging3x3(Iyy, image_width, image_height)
    Ixy = IPSmooth.computeGaussianAveraging3x3(Ixy, image_width, image_height)

    # 4) 计算 Harris 响应 R = det(M) - k*(trace(M))^2
    R_vals = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(image_height):
        for c in range(image_width):
            detM = Ixx[r][c] * Iyy[r][c] - Ixy[r][c] * Ixy[r][c]
            traceM = Ixx[r][c] + Iyy[r][c]
            R_vals[r][c] = detM - k * (traceM ** 2)

    # 5) 使用相对阈值进行角点初筛
    maxR = max(R_vals[r][c] for r in range(image_height) for c in range(image_width))
    threshold = threshold_ratio * maxR  # 阈值 = 阈值比例 * 最大响应值
    corners = []
    for r in range(image_height):
        for c in range(image_width):
            if R_vals[r][c] > threshold:
                corners.append((c, r))  # (x, y) 格式

    if not do_nonmax_suppression:
        return corners

    # 6) 非极大值抑制：在 3x3 邻域内保留局部最大值
    final_corners = []
    window_size = 3
    offset = window_size // 2
    for (cx, cy) in corners:
        r, c = cy, cx
        current_val = R_vals[r][c]
        is_max = True
        for rr in range(-offset, offset + 1):
            for cc in range(-offset, offset + 1):
                nr = r + rr
                nc = c + cc
                if nr < 0 or nr >= image_height or nc < 0 or nc >= image_width:
                    continue
                if R_vals[nr][nc] > current_val:
                    is_max = False
                    break
            if not is_max:
                break
        if is_max:
            final_corners.append((cx, cy))
    return final_corners



# 原有的辅助函数：生成 RGB 图像及拼接图,这之后是我们的项目源代码，没咋改了

def prepareRGBImageFromIndividualArrays(r_pixel_array, g_pixel_array, b_pixel_array, image_width, image_height):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = [r_pixel_array[y][x], g_pixel_array[y][x], b_pixel_array[y][x]]
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):
    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]
    return matchingImage



# 主函数：图像读取、预处理、角点检测及可视化

def main():
    filename_left_image = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"


    (image_width, image_height, px_array_left_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(
        filename_left_image)
    (image_width, image_height, px_array_right_original) = IORW.readRGBImageAndConvertToGreyscalePixelArray(
        filename_right_image)

    start = timer()
    px_array_left = IPSmooth.computeGaussianAveraging3x3(px_array_left_original, image_width, image_height)
    px_array_right = IPSmooth.computeGaussianAveraging3x3(px_array_right_original, image_width, image_height)
    end = timer()
    print("elapsed time image smoothing: ", end - start)

    # 将灰度图拉伸到 0~255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)

    # Harris 角点检测
    harris_corners_left = computeHarrisCorners(px_array_left, image_width, image_height,
                                               k=0.05, threshold_ratio=0.01, do_nonmax_suppression=True)
    harris_corners_right = computeHarrisCorners(px_array_right, image_width, image_height,
                                                k=0.05, threshold_ratio=0.01, do_nonmax_suppression=True)
    print(f"Detected {len(harris_corners_left)} corners in left image.")
    print(f"Detected {len(harris_corners_right)} corners in right image.")

    # 可视化 1：显示左右图像及其检测的 Harris 角点
    fig1, axs1 = pyplot.subplots(1, 2, figsize=(10, 5))
    axs1[0].imshow(px_array_left, cmap='gray')
    axs1[0].set_title("Left image with Harris corners")
    for (x, y) in harris_corners_left:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs1[0].add_patch(circ)

    axs1[1].imshow(px_array_right, cmap='gray')
    axs1[1].set_title("Right image with Harris corners")
    for (x, y) in harris_corners_right:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs1[1].add_patch(circ)

    pyplot.show()

    # 可视化 2：拼接图像并绘制左右图中心连接线
    matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)
    pyplot.imshow(matchingImage, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matching image (center connection)")
    pointA = (image_width / 2, image_height / 2)
    pointB = (3 * image_width / 2, image_height / 2)
    connection = ConnectionPatch(pointA, pointB, "data", edgecolor='r', linewidth=1)
    ax.add_artist(connection)
    pyplot.show()


if __name__ == "__main__":
    main()
