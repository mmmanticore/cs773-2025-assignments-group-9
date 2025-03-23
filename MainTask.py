import random
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
from scipy.signal import convolve2d

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils


# 1) 3×3 Averaging Filter (Initial Smoothing)

def computeAveraging3x3(px_array, image_width, image_height):
    """
    Apply 3×3 averaging filter (box smoothing) to the input image.
    """
    out = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    # If the skeleton code skips edges, change to range(1, image_height-1)
    for r in range(1, image_height - 1):
        for c in range(1, image_width - 1):
            s = 0.0
            for rr in range(-1, 2):
                for cc in range(-1, 2):
                    s += px_array[r + rr][c + cc]
            out[r][c] = s / 9.0
    return out


# 2) 5×5 Gaussian Filter (sigma=1) - Used for smoothing Ixx, Iyy, Ixy

def computeGaussianAveraging_sigma1(pixel_array, image_width, image_height, kernel_size=5):
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * 1.0**2))
    kernel /= kernel.sum()

    img = np.array(pixel_array, dtype=np.float64)
    smoothed = convolve2d(img, kernel, mode='same', boundary='symm')
    return smoothed.tolist()


# 3) Sobel Gradient

def computePartialDerivativesSobel(px_array, image_width, image_height):
    """
    Apply 3x3 Sobel filter to the entire image. If matching the skeleton, compute only within [1, width-1).
    """
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    sobel_y = [[-1, -2, -1],
               [0,  0,  0],
               [1,  2,  1]]

    Ix = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    Iy = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)

    for r in range(image_height):
        for c in range(image_width):
            gx = 0.0
            gy = 0.0
            for rr in range(-1, 2):
                for cc in range(-1, 2):
                    nr = r + rr
                    nc = c + cc
                    if 0 <= nr < image_height and 0 <= nc < image_width:
                        pixel_val = px_array[nr][nc]
                        gx += pixel_val * sobel_x[rr + 1][cc + 1]
                        gy += pixel_val * sobel_y[rr + 1][cc + 1]
            Ix[r][c] = gx
            Iy[r][c] = gy
    return Ix, Iy


# 4) Harris Corner Detection: Force return top 1000

def computeHarrisCorners(px_array, image_width, image_height, k=0.04):
    # 1) Compute gradients
    Ix, Iy = computePartialDerivativesSobel(px_array, image_width, image_height)

    # 2) Compute Ixx, Iyy, Ixy
    Ixx = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    Iyy = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    Ixy = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(image_height):
        for c in range(image_width):
            Ixx[r][c] = Ix[r][c] * Ix[r][c]
            Iyy[r][c] = Iy[r][c] * Iy[r][c]
            Ixy[r][c] = Ix[r][c] * Iy[r][c]

    # 3) Apply 5×5 Gaussian smoothing
    Ixx = computeGaussianAveraging_sigma1(Ixx, image_width, image_height, kernel_size=5)
    Iyy = computeGaussianAveraging_sigma1(Iyy, image_width, image_height, kernel_size=5)
    Ixy = computeGaussianAveraging_sigma1(Ixy, image_width, image_height, kernel_size=5)

    # 4) Compute Harris response
    R_vals = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(image_height):
        for c in range(image_width):
            detM = Ixx[r][c] * Iyy[r][c] - Ixy[r][c] * Ixy[r][c]
            traceM = Ixx[r][c] + Iyy[r][c]
            R_vals[r][c] = detM - k * (traceM**2)

    # 5) Collect candidate corners with threshold=1e7
    threshold = 1e7
    # Can be changed to >= threshold
    candidates = []
    for r in range(image_height):
        for c in range(image_width):
            if R_vals[r][c] >= threshold:
                candidates.append(((c, r), R_vals[r][c]))

    # If fewer than 1000, relax the threshold (i.e., collect all pixels)
    if len(candidates) < 1000:
        candidates = []
        for r in range(image_height):
            for c in range(image_width):
                candidates.append(((c, r), R_vals[r][c]))

    # 6) Apply 3×3 non-maximum suppression
    local_maxima = []
    for ((cx, cy), val) in candidates:
        is_max = True
        for rr in range(-1, 2):
            for cc in range(-1, 2):
                nr = cy + rr
                nc = cx + cc
                if 0 <= nr < image_height and 0 <= nc < image_width:
                    if (nr, nc) != (cy, cx) and R_vals[nr][nc] > val:
                        is_max = False
                        break
            if not is_max:
                break
        if is_max:
            local_maxima.append(((cx, cy), val))

    # 7) Sort by response value in descending order, take top 1000
    local_maxima.sort(key=lambda x: x[1], reverse=True)
    final_corners = [pt for (pt, _) in local_maxima[:1000]]
    print(f"Harris corners found = {len(local_maxima)}; returning top {len(final_corners)}")
    return final_corners


# 5) Feature Descriptors & Matching

def computeFeatureDescriptors(px_array, corners, image_width, image_height, window_size=7):
    half_w = window_size // 2
    descriptors = []
    for (x, y) in corners:
        patch_pixels = []
        for rr in range(y - half_w, y + half_w + 1):
            for cc in range(x - half_w, x + half_w + 1):
                if 0 <= rr < image_height and 0 <= cc < image_width:
                    patch_pixels.append(px_array[rr][cc])
                else:
                    patch_pixels.append(0)
        mean_val = sum(patch_pixels) / float(len(patch_pixels))
        sum_sq_diff = sum((val - mean_val)**2 for val in patch_pixels)
        std_val = (sum_sq_diff / len(patch_pixels))**0.5
        if std_val < 1e-12:
            std_val = 1.0
        desc = [(val - mean_val) / std_val for val in patch_pixels]
        descriptors.append(desc)
    return descriptors

def matchFeatures(descriptors_left, descriptors_right, corners_left, corners_right, distance_threshold=0.6):
    matches = []
    for i, descL in enumerate(descriptors_left):
        dists = []
        for j, descR in enumerate(descriptors_right):
            dist = np.linalg.norm(np.array(descL) - np.array(descR))
            dists.append((dist, j))
        dists.sort(key=lambda x: x[0])
        if len(dists) >= 2:
            nearest, second_nearest = dists[0], dists[1]
            if nearest[0] < distance_threshold * second_nearest[0]:
                matches.append((corners_left[i], corners_right[nearest[1]]))
    return matches

def computeHomography(match_pairs):
    A = []
    for (xL, yL), (xR, yR) in match_pairs:
        A.append([-xL, -yL, -1, 0, 0, 0, xL*xR, yL*xR, xR])
        A.append([0, 0, 0, -xL, -yL, -1, xL*yR, yL*yR, yR])
    A = np.array(A)
    _, _, vh = np.linalg.svd(A)
    h = vh[-1, :] / vh[-1, -1]
    return h.reshape((3, 3))

def ransacHomography(matches, num_iterations=1000, inlier_threshold=5.0):
    if len(matches) < 4:
        return None
    best_H = None
    best_inliers = 0
    for _ in range(num_iterations):
        sample = random.sample(matches, 4)
        Hcand = computeHomography(sample)
        inliers_count = 0
        for (xL, yL), (xR, yR) in matches:
            denom = Hcand[2][0]*xL + Hcand[2][1]*yL + Hcand[2][2]
            if abs(denom) < 1e-12:
                continue
            px = (Hcand[0][0]*xL + Hcand[0][1]*yL + Hcand[0][2]) / denom
            py = (Hcand[1][0]*xL + Hcand[1][1]*yL + Hcand[1][2]) / denom
            dx = px - xR
            dy = py - yR
            if (dx*dx + dy*dy) < (inlier_threshold*inlier_threshold):
                inliers_count += 1
        if inliers_count > best_inliers:
            best_inliers = inliers_count
            best_H = Hcand

    if best_H is None:
        return None

    # Optional: Collect inliers and recompute H with more points
    inliers = []
    for (xL, yL), (xR, yR) in matches:
        denom = best_H[2][0]*xL + best_H[2][1]*yL + best_H[2][2]
        if abs(denom) < 1e-12:
            continue
        px = (best_H[0][0]*xL + best_H[0][1]*yL + best_H[0][2]) / denom
        py = (best_H[1][0]*xL + best_H[1][1]*yL + best_H[1][2]) / denom
        dx = px - xR
        dy = py - yR
        if (dx*dx + dy*dy) < (inlier_threshold*inlier_threshold):
            inliers.append(((xL, yL), (xR, yR)))
    if len(inliers) >= 4:
        best_H = computeHomography(inliers)
    return best_H


# Stitching & Visualization

from scipy.ndimage import distance_transform_edt as edt

def gaussian_blend(dist_left, dist_right, sigma=10):
    from scipy.ndimage import gaussian_filter
    blended_weight = gaussian_filter(dist_left - dist_right, sigma=sigma)
    mn, mx = blended_weight.min(), blended_weight.max()
    if abs(mx - mn) < 1e-12:
        return 0.5 * np.ones_like(blended_weight)
    return (blended_weight - mn) / (mx - mn)

def crop_black_borders(img):
    arr = np.array(img)
    mask = arr > 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        return arr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return arr[y0:y1, x0:x1]

def warpAndStitch(px_array_left, px_array_right, H_left_to_right, image_width, image_height):
    corners_right = np.array([[0, 0, 1],
                              [image_width, 0, 1],
                              [image_width, image_height, 1],
                              [0, image_height, 1]]).T
    corners_left = np.array([[0, 0, 1],
                             [image_width, 0, 1],
                             [image_width, image_height, 1],
                             [0, image_height, 1]]).T
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

    rightCanvas = np.zeros((out_h, out_w), dtype=np.float64)
    leftCanvas  = np.zeros((out_h, out_w), dtype=np.float64)
    rightMask   = np.zeros((out_h, out_w), dtype=np.float64)
    leftMask    = np.zeros((out_h, out_w), dtype=np.float64)

    # Copy right image
    for r in range(image_height):
        for c in range(image_width):
            yy = r + offset_y
            xx = c + offset_x
            if 0 <= yy < out_h and 0 <= xx < out_w:
                rightCanvas[yy, xx] = px_array_right[r][c]
                rightMask[yy, xx]   = 1

    # Forward map left image
    for r in range(image_height):
        for c in range(image_width):
            vec = H_left_to_right @ np.array([c, r, 1], dtype=np.float64)
            w_  = vec[2]
            if abs(w_) < 1e-12:
                continue
            x_mapped = vec[0] / w_
            y_mapped = vec[1] / w_
            out_x = x_mapped + offset_x
            out_y = y_mapped + offset_y
            ix = int(round(out_x))
            iy = int(round(out_y))
            if 0 <= ix < out_w and 0 <= iy < out_h:
                leftCanvas[iy, ix] = px_array_left[r][c]
                leftMask[iy, ix]   = 1

    dist_left  = edt(leftMask)
    dist_right = edt(rightMask)
    alpha = gaussian_blend(dist_left, dist_right, sigma=20)

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
    return crop_black_borders(stitched)

def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):
    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width*2, image_height)
    for r in range(image_height):
        for c in range(image_width):
            matchingImage[r][c] = left_pixel_array[r][c]
            matchingImage[r][c + image_width] = right_pixel_array[r][c]
    return matchingImage


# main

def main():
    filename_left_image  = "./images/panoramaStitching/tongariro_left_01.png"
    filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"

    (image_width, image_height, px_left_orig)  = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_left_image)
    (image_width, image_height, px_right_orig) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_right_image)

    # 1) 3×3 均值滤波：初始模糊
    px_left_blurred  = computeAveraging3x3(px_left_orig,  image_width, image_height)
    px_right_blurred = computeAveraging3x3(px_right_orig, image_width, image_height)

    # 2) Stretch 到 [0,255]，量化
    px_left  = IPPixelOps.scaleTo0And255AndQuantize(px_left_blurred,  image_width, image_height)
    px_right = IPPixelOps.scaleTo0And255AndQuantize(px_right_blurred, image_width, image_height)

    # 3) Harris 角点检测（按老师要求保证返回1000个角点）
    corners_left  = computeHarrisCorners(px_left,  image_width, image_height, k=0.04)
    corners_right = computeHarrisCorners(px_right, image_width, image_height, k=0.04)

    def save_corners_to_csv(corners, output_path):
        with open(output_path, 'w') as f:
            f.write("Index,X,Y\n")
            for idx, (x, y) in enumerate(corners):
                f.write(f"{idx},{x},{y}\n")

    save_corners_to_csv(corners_left,  "left_image_corners.csv")
    save_corners_to_csv(corners_right, "right_image_corners.csv")

    print(f"Detected {len(corners_left)} corners in left image.")
    print(f"Detected {len(corners_right)} corners in right image.")

    # 4) Compute descriptors
    window_size = 7
    desc_left  = computeFeatureDescriptors(px_left,  corners_left,  image_width, image_height, window_size)
    desc_right = computeFeatureDescriptors(px_right, corners_right, image_width, image_height, window_size)

    print(f"Descriptors (left) : {len(desc_left)}")
    print(f"Descriptors (right): {len(desc_right)}")

    # 5) 特征匹配
    # 将匹配比率阈值从0.75调到0.85，以便获得更多匹配
    matches = matchFeatures(desc_left, desc_right, corners_left, corners_right, distance_threshold=0.85)
    print(f"Found {len(matches)} raw matches.")

    # 6) RANSAC：将 inlier_threshold 调整为2.0，迭代次数增加到10000
    best_H = ransacHomography(matches, num_iterations=10000, inlier_threshold=2.0)
    if best_H is None:
        print("RANSAC failed: no homography found.")
        return
    print("Estimated homography (H):\n", best_H)

    # 7) 拼接
    stitched = warpAndStitch(px_left, px_right, best_H, image_width, image_height)
    stitched = crop_black_borders(stitched)

    # 8) 匹配可视化
    matchingImage = prepareMatchingImage(px_left, px_right, image_width, image_height)

    fig, axs = pyplot.subplots(1, 3, figsize=(15, 5))
    # 左图 + 角点
    axs[0].imshow(px_left, cmap='gray')
    axs[0].set_title("Left with corners")
    for (x, y) in corners_left:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs[0].add_patch(circ)
    # 右图 + 角点
    axs[1].imshow(px_right, cmap='gray')
    axs[1].set_title("Right with corners")
    for (x, y) in corners_right:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs[1].add_patch(circ)
    # 匹配可视化
    axs[2].imshow(matchingImage, cmap='gray')
    axs[2].set_title("Matches (green lines)")
    for ((xL, yL), (xR, yR)) in matches:
        xR_pano = xR + image_width
        line = ConnectionPatch((xL, yL), (xR_pano, yR), "data", edgecolor='g', linewidth=0.5)
        axs[2].add_artist(line)
    pyplot.show()

    # 9) 拼接结果展示
    pyplot.imshow(stitched, cmap='gray')
    pyplot.title("Stitched Result")
    pyplot.show()

if __name__ == "__main__":
    main()
