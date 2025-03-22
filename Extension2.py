import math

import random
import numpy as np
from PIL import Image
import imageIO.png
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
from timeit import default_timer as timer
from scipy.ndimage import distance_transform_edt as edt, binary_dilation
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
    Ixx = IPSmooth.computeGaussianAveraging(Ixx, image_width, image_height, kernel_size=7)
    Iyy = IPSmooth.computeGaussianAveraging(Iyy, image_width, image_height, kernel_size=7)
    Ixy = IPSmooth.computeGaussianAveraging(Ixy, image_width, image_height, kernel_size=7)

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


# 灰度转换
def loadImageasGreyscaleArray(image_path):
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


def computeFeatureDescriptors(px_array, corners, image_width, image_height, window_size=7):
    """
    英文注释 (English):
    ------------------
    This function computes a simple patch-based feature descriptor for each corner.
    1) For each corner (x, y), we extract a (window_size × window_size) patch centered on (x, y).
    2) Flatten this patch into a 1D vector.
    3) Normalize it by subtracting mean and dividing by standard deviation (or 1 if std=0).
    4) Return a list of descriptors, each descriptor is a list (or vector) of length (window_size*window_size).

    中文注释 (Chinese):
    ------------------
    此函数为每个角点计算一个基于图像块（patch）的简单特征描述子。
    1) 对于每个角点 (x, y)，在其中心提取一个 (window_size × window_size) 的图像块。
    2) 将该图像块展平为一维向量。
    3) 进行归一化：减去像素均值，再除以像素标准差（若标准差为0则除以1）。
    4) 返回一个描述子列表，每个描述子是长度为 (window_size*window_size) 的一维列表。

    参数 (Parameters):
    -----------------
    px_array     : 2D list (height × width) - the grayscale image pixel array.
    corners      : list of (x, y) tuples - the corner coordinates.
    image_width  : int - width of the image.
    image_height : int - height of the image.
    window_size  : int - size of the patch around each corner, must be odd (e.g. 7).

    返回 (Returns):
    --------------
    descriptors : list of lists
        Each element corresponds to one corner, and is the normalized patch flattened into a 1D list.
        The i-th descriptor corresponds to the i-th corner in the input 'corners' list.
    """

    half_w = window_size // 2
    descriptors = []

    for (x, y) in corners:
        # 1) Extract the patch around (x, y)
        #    确保不越界，如果角点在边缘附近，窗口要截断
        patch_pixels = []
        for rr in range(y - half_w, y + half_w + 1):
            for cc in range(x - half_w, x + half_w + 1):
                if rr < 0 or rr >= image_height or cc < 0 or cc >= image_width:
                    # 越界则填充0或其他值
                    patch_pixels.append(0)
                else:
                    patch_pixels.append(px_array[rr][cc])

        # 2) 计算该 patch 的均值和标准差
        patch_size = len(patch_pixels)
        mean_val = sum(patch_pixels) / float(patch_size)

        # 计算标准差
        sum_sq_diff = 0.0
        for val in patch_pixels:
            diff = val - mean_val
            sum_sq_diff += diff * diff
        std_val = (sum_sq_diff / patch_size) ** 0.5
        if std_val == 0.0:
            std_val = 1.0  # 避免除0

        # 3) 归一化: (pixel - mean) / std
        normalized_patch = []
        for val in patch_pixels:
            norm_val = (val - mean_val) / std_val
            normalized_patch.append(norm_val)

        # 4) 将该 patch 的归一化结果作为描述子
        descriptors.append(normalized_patch)

    return descriptors

# 4. 特征匹配 (Naive 最近邻 + 阈值)
def matchFeatures(descriptors_left, descriptors_right, corners_left, corners_right, distance_threshold=0.6):
    matches = []
    for i, descL in enumerate(descriptors_left):
        distances = []
        for j, descR in enumerate(descriptors_right):
            dist = np.linalg.norm(np.array(descL) - np.array(descR))
            distances.append((dist, j))
        distances.sort(key=lambda x: x[0])
        if len(distances) >= 2:
            nearest, second_nearest = distances[0], distances[1]
            if nearest[0] < distance_threshold * second_nearest[0]:
                matches.append((corners_left[i], corners_right[nearest[1]]))
    return matches


# 5. 计算单应矩阵 (Homography)

def computeHomography(match_pairs):
    A = []
    for (xL, yL), (xR, yR) in match_pairs:
        A.append([-xL, -yL, -1, 0, 0, 0, xL*xR, yL*xR, xR])
        A.append([0, 0, 0, -xL, -yL, -1, xL*yR, yL*yR, yR])

    A = np.array(A)
    _, _, vh = np.linalg.svd(A)
    h = vh[-1, :] / vh[-1, -1]  # 标准化
    H = h.reshape((3, 3))
    return H

# 6. RANSAC to find best Homography

def ransacHomography(matches, num_iterations=1000, inlier_threshold=5.0):
    """
    在原基础上增加:
    1) RANSAC 找到 best_H.
    2) 用 best_H 的内点再次用全部内点计算一次 Homography (可选)，提高精度。
    """
    if len(matches)<4:
        return None
    best_H=None
    best_inliers=0
    import math

    # 1) 常规 RANSAC
    for _ in range(num_iterations):
        sample=random.sample(matches,4)
        Hcand=computeHomography(sample)
        inliers_count=0
        for (xL,yL),(xR,yR) in matches:
            denom=Hcand[2][0]*xL+Hcand[2][1]*yL+Hcand[2][2]
            if abs(denom)<1e-12:
                continue
            px=(Hcand[0][0]*xL+Hcand[0][1]*yL+Hcand[0][2])/denom
            py=(Hcand[1][0]*xL+Hcand[1][1]*yL+Hcand[1][2])/denom
            dx=px-xR
            dy=py-yR
            if (dx*dx+dy*dy)<(inlier_threshold*inlier_threshold):
                inliers_count+=1
        if inliers_count>best_inliers:
            best_inliers=inliers_count
            best_H=Hcand

    if best_H is None:
        return None

    # 2) 收集 best_H 的内点，再次计算 H (可选)
    inliers=[]
    for (xL,yL),(xR,yR) in matches:
        denom=best_H[2][0]*xL+best_H[2][1]*yL+best_H[2][2]
        if abs(denom)<1e-12:
            continue
        px=(best_H[0][0]*xL+best_H[0][1]*yL+best_H[0][2])/denom
        py=(best_H[1][0]*xL+best_H[1][1]*yL+best_H[1][2])/denom
        dx=px-xR
        dy=py-yR
        if (dx*dx+dy*dy)<(inlier_threshold*inlier_threshold):
            inliers.append(((xL,yL),(xR,yR)))

    if len(inliers)>=4:
        best_H=computeHomography(inliers)

    return best_H


# 7. 图像拼接：warp 右图到左图坐标系并合并

def bilinear_interpolate(img, x, y):
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    if x0 < 0 or x1 >= len(img[0]) or y0 < 0 or y1 >= len(img):
        return 0
    a, b = x - x0, y - y0
    return (img[y0][x0] * (1 - a) * (1 - b) +
            img[y0][x1] * a * (1 - b) +
            img[y1][x0] * (1 - a) * b +
            img[y1][x1] * a * b)

def gaussian_blend(dist_left, dist_right, sigma=10):
    from scipy.ndimage import gaussian_filter
    # 计算 (dist_left - dist_right) 的高斯平滑结果
    blended_weight = gaussian_filter(dist_left - dist_right, sigma=sigma)
    # 归一化到 [0, 1]
    mn, mx = blended_weight.min(), blended_weight.max()
    if abs(mx - mn) < 1e-12:
        # 避免分母为0的情况
        return 0.5 * np.ones_like(blended_weight)
    alpha = (blended_weight - mn) / (mx - mn)
    return alpha

def crop_black_borders(img):
    """从拼接结果中裁剪掉纯黑的区域。"""
    arr = np.array(img)
    mask = arr > 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        return arr  # 整张图都没内容就直接返回
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = arr[y0:y1, x0:x1]
    return cropped

def warpAndStitch(px_array_left, px_array_right, H_left_to_right, image_width, image_height):
    """
    保持右图坐标系不变，将左图前向映射到右图坐标系后做融合。
    1) 计算“右图固定”+“左图映射后”在输出图上的外包框大小及偏移量
    2) 分配画布，并把右图原地拷贝过去（不用变换）
    3) 前向映射左图：对左图每个像素，用 H_left_to_right 求它在右图坐标系的位置
    4) 做距离变换 + 高斯混合得到 alpha
    5) 在重叠区用 alpha 融合，然后裁剪黑边返回
    """



    #1. 计算输出画布大小及偏移

    # 1.1 右图的四个角点 (保持原地不动)
    corners_right = np.array([
        [0, 0, 1],
        [image_width, 0, 1],
        [image_width, image_height, 1],
        [0, image_height, 1]
    ]).T

    # 1.2 左图的四个角点，使用 H_left_to_right 前向映射到右图坐标系
    corners_left = np.array([
        [0, 0, 1],
        [image_width, 0, 1],
        [image_width, image_height, 1],
        [0, image_height, 1]
    ]).T

    mapped_left = H_left_to_right @ corners_left
    mapped_left /= mapped_left[2, :]  # 齐次坐标归一化

    # 将“右图角点坐标”和“映射后的左图角点坐标”合并，找最小/最大 x,y
    xs = np.concatenate((corners_right[0, :], mapped_left[0, :]))
    ys = np.concatenate((corners_right[1, :], mapped_left[1, :]))

    min_x, max_x = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    min_y, max_y = int(np.floor(ys.min())), int(np.ceil(ys.max()))

    out_w = max_x - min_x
    out_h = max_y - min_y
    offset_x = -min_x
    offset_y = -min_y

    #2. 分配画布并分别绘制“右图原地”和“左图前向映射”
    rightCanvas = np.zeros((out_h, out_w), dtype=np.float64)
    leftCanvas  = np.zeros((out_h, out_w), dtype=np.float64)
    rightMask   = np.zeros((out_h, out_w), dtype=np.float64)
    leftMask    = np.zeros((out_h, out_w), dtype=np.float64)

    # 2.1 将右图“固定”拷贝到输出画布中
    for r in range(image_height):
        for c in range(image_width):
            yy = r + offset_y
            xx = c + offset_x
            if 0 <= yy < out_h and 0 <= xx < out_w:
                rightCanvas[yy, xx] = px_array_right[r][c]
                rightMask[yy, xx]   = 1

    # 2.2前向映射左图到右图坐标系 (最近邻)
    for r in range(image_height):
        for c in range(image_width):
            # 用 H_left_to_right 映射 (c,r) 到右图坐标系
            vec = H_left_to_right @ np.array([c, r, 1], dtype=np.float64)
            w_  = vec[2]
            if abs(w_) < 1e-12:
                continue
            x_mapped = vec[0] / w_
            y_mapped = vec[1] / w_

            # 再加 offset 放到输出画布上
            out_x = x_mapped + offset_x
            out_y = y_mapped + offset_y

            # 最近邻插值 (也可改成 round+check 或 bilinear)
            ix = int(round(out_x))
            iy = int(round(out_y))
            if 0 <= ix < out_w and 0 <= iy < out_h:
                leftCanvas[iy, ix] = px_array_left[r][c]
                leftMask[iy, ix]   = 1

    #  3.距离变换 + 高斯融合权重 alpha

    dist_left  = edt(leftMask)
    dist_right = edt(rightMask)
    alpha = gaussian_blend(dist_left, dist_right, sigma=20)

    # 4. 最终融合 (alpha * left + (1-alpha) * right)

    stitched = np.zeros((out_h, out_w), dtype=np.float64)
    for sy in range(out_h):
        for sx in range(out_w):
            lv = leftCanvas[sy, sx]
            rv = rightCanvas[sy, sx]
            lm = leftMask[sy, sx]
            rm = rightMask[sy, sx]
            if lm > 0 and rm > 0:
                a = alpha[sy, sx]
                stitched[sy, sx] = a * lv + (1 - a) * rv
            elif lm > 0:
                stitched[sy, sx] = lv
            elif rm > 0:
                stitched[sy, sx] = rv
            else:
                stitched[sy, sx] = 0

    # 5.裁剪黑边并返回

    return crop_black_borders(stitched)

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

def filter_corners(corners, image_width, image_height, border=20):
    # 剔除靠近图像边缘的角点
    filtered = []
    for (x, y) in corners:
        if border < x < (image_width - border) and border < y < (image_height - border):
            filtered.append((x, y))
    return filtered

def warpAndStitchInverse(px_array_left, px_array_right, H_left_to_right, image_width, image_height):
    """
    额外的拼接函数：使用“反向映射 + 双线性插值”来消除网格和空洞。
    思路：
      1) 先确定输出画布大小 & 偏移量，与原先类似；
      2) 右图保持不变，直接拷贝；
      3) 遍历输出画布 (sx, sy)，用 invH = (H_left_to_right)^(-1) 找到左图坐标 (xL, yL)，双线性插值得到左图像素值；
      4) 做距离变换 & 高斯融合；
      5) 裁剪黑边并返回。
    """

    corners_right = np.array([
        [0, 0, 1],
        [image_width, 0, 1],
        [image_width, image_height, 1],
        [0, image_height, 1]
    ]).T

    corners_left = np.array([
        [0, 0, 1],
        [image_width, 0, 1],
        [image_width, image_height, 1],
        [0, image_height, 1]
    ]).T

    # 用 H_left_to_right 映射左图四角，确定 bounding box
    mapped_left = H_left_to_right @ corners_left
    mapped_left /= mapped_left[2, :]

    xs = np.concatenate((corners_right[0, :], mapped_left[0, :]))
    ys = np.concatenate((corners_right[1, :], mapped_left[1, :]))

    min_x, max_x = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    min_y, max_y = int(np.floor(ys.min())), int(np.ceil(ys.max()))

    out_w = max_x - min_x
    out_h = max_y - min_y
    offset_x = -min_x
    offset_y = -min_y

    #2. 分配画布
    rightCanvas = np.zeros((out_h, out_w), dtype=np.float64)
    leftCanvas = np.zeros((out_h, out_w), dtype=np.float64)
    rightMask = np.zeros((out_h, out_w), dtype=np.float64)
    leftMask = np.zeros((out_h, out_w), dtype=np.float64)

    # 3.右图保持不变，直接拷贝
    for r in range(image_height):
        for c in range(image_width):
            yy = r + offset_y
            xx = c + offset_x
            if 0 <= yy < out_h and 0 <= xx < out_w:
                rightCanvas[yy, xx] = px_array_right[r][c]
                rightMask[yy, xx] = 1

    # 4.左图使用“反向映射 + 双线性插值”
    invH = np.linalg.inv(H_left_to_right)

    def bilinearInterp(px_array, x, y):
        # 双线性插值，x,y为浮点坐标
        x0 = int(math.floor(x))
        x1 = x0 + 1
        y0 = int(math.floor(y))
        y1 = y0 + 1
        if x0 < 0 or x1 >= image_width or y0 < 0 or y1 >= image_height:
            return 0  # 越界就返回0
        dx = x - x0
        dy = y - y0
        val00 = px_array[y0][x0]
        val01 = px_array[y0][x1]
        val10 = px_array[y1][x0]
        val11 = px_array[y1][x1]
        return ((val00 * (1 - dx) * (1 - dy)) +
                (val01 * dx * (1 - dy)) +
                (val10 * (1 - dx) * dy) +
                (val11 * dx * dy))

    for sy in range(out_h):
        for sx in range(out_w):
            vec = invH @ np.array([sx - offset_x, sy - offset_y, 1], dtype=np.float64)
            w_ = vec[2]
            if abs(w_) < 1e-12:
                continue
            xL = vec[0] / w_
            yL = vec[1] / w_

            # 不要再用 "if val>0" 判断，直接接收插值结果
            val = bilinearInterp(px_array_left, xL, yL)
            # 如果插值在左图有效范围内，就写入 leftCanvas
            if val != 0:
                leftCanvas[sy, sx] = val
                leftMask[sy, sx] = 1

    # 5.做距离变换 + 高斯融合
    # 5.1 先对 mask 做形态学膨胀，减少锐利边缘
    # 根据需要调节 iterations=1/2/3
    leftMask = binary_dilation(leftMask, iterations=2).astype(np.float64)
    rightMask = binary_dilation(rightMask, iterations=2).astype(np.float64)

    dist_left = edt(leftMask)
    dist_right = edt(rightMask)

    # 把高斯融合的 sigma 调大，融合带更宽
    alpha = gaussian_blend(dist_left, dist_right, sigma=35)

    stitched = np.zeros((out_h, out_w), dtype=np.float64)
    for sy in range(out_h):
        for sx in range(out_w):
            lv = leftCanvas[sy, sx]
            rv = rightCanvas[sy, sx]
            lm = leftMask[sy, sx]
            rm = rightMask[sy, sx]
            if lm > 0 and rm > 0:
                a = alpha[sy, sx]
                stitched[sy, sx] = a * lv + (1 - a) * rv
            elif lm > 0:
                stitched[sy, sx] = lv
            elif rm > 0:
                stitched[sy, sx] = rv
            else:
                stitched[sy, sx] = 0

    # 裁剪黑边并返回
    return crop_black_borders(stitched)

def main():

    filename_left_image = "./images/panoramaStitching/2.1.jpg"
    filename_right_image = "./images/panoramaStitching/2.2.jpg"

    (image_width, image_height, px_array_left_original) = loadImageasGreyscaleArray(filename_left_image)
    (image_width, image_height, px_array_right_original) = loadImageasGreyscaleArray(filename_right_image)

    start = timer()
    px_array_left = IPSmooth.computeGaussianAveraging(px_array_left_original, image_width, image_height ,kernel_size=7)
    px_array_right = IPSmooth.computeGaussianAveraging(px_array_right_original, image_width, image_height, kernel_size=7)
    end = timer()
    print("elapsed time image smoothing: ", end - start)

    # Stretch to 0~255
    px_array_left = IPPixelOps.scaleTo0And255AndQuantize(px_array_left, image_width, image_height)
    px_array_right = IPPixelOps.scaleTo0And255AndQuantize(px_array_right, image_width, image_height)

    # 1.Harris corners
    # 调整 threshold_ratio
    harris_corners_left = computeHarrisCorners(px_array_left, image_width, image_height,
                                               k=0.05, threshold_ratio=0.025, do_nonmax_suppression=True)
    harris_corners_right = computeHarrisCorners(px_array_right, image_width, image_height,
                                                k=0.05, threshold_ratio=0.025, do_nonmax_suppression=True)

    # 过滤边缘角点
    harris_corners_left = filter_corners(harris_corners_left, image_width, image_height)
    harris_corners_right = filter_corners(harris_corners_right, image_width, image_height)

    print(f"Detected {len(harris_corners_left)} corners in left image.")
    print(f"Detected {len(harris_corners_right)} corners in right image.")

    # 2.Compute descriptors
    window_size = 7
    descriptors_left = computeFeatureDescriptors(px_array_left, harris_corners_left,
                                                 image_width, image_height, window_size)
    descriptors_right = computeFeatureDescriptors(px_array_right, harris_corners_right,
                                                  image_width, image_height, window_size)

    print(f"Number of descriptors in left image = {len(descriptors_left)}")
    print(f"Number of descriptors in right image = {len(descriptors_right)}")

    # 3. Feature matching
    # Adjust distance_threshold if needed
    matches = matchFeatures(descriptors_left, descriptors_right,
                            harris_corners_left, harris_corners_right,
                            distance_threshold=0.75)
    print(f"Found {len(matches)} raw matches.")

    # 4.RANSAC to estimate homography
    best_H = ransacHomography(matches, num_iterations=5000, inlier_threshold=1.0)
    if best_H is None:
        print("RANSAC failed: no homography found.")
        return

    print("Estimated homography (H):")
    for row in best_H:
        print(row)

    # 5.Warp & stitch
    stitched = warpAndStitch(px_array_left, px_array_right, best_H, image_width, image_height)
    stitched = crop_black_borders(stitched)
    stitched_inverse = warpAndStitchInverse(px_array_left, px_array_right, best_H,
                                            image_width, image_height)
    # 6.可视化
    # 6.1.原图上画角点
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

    #6.2. 匹配可视化 (简单并排)
    # 先拼接左右图
    matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)
    pyplot.imshow(matchingImage, cmap='gray')
    ax = pyplot.gca()
    ax.set_title("Matches (green lines)")

    for ((xL, yL), (xR, yR)) in matches:
        # xR 在拼接图中要加偏移
        xR_pano = xR + image_width
        line = ConnectionPatch((xL, yL), (xR_pano, yR), "data", edgecolor='g', linewidth=0.5)
        ax.add_artist(line)

    pyplot.show()

    # 6.3.显示拼接结果
    pyplot.imshow(stitched, cmap='gray')
    pyplot.title("Stitched Result")
    pyplot.show()
    pyplot.imshow(stitched_inverse, cmap='gray')
    pyplot.title("Stitched Result (Inverse Warping)")
    pyplot.show()

if __name__ == "__main__":
    main()