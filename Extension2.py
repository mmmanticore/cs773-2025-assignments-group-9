import random
import numpy as np
from PIL import Image
from matplotlib import pyplot
from matplotlib.patches import Circle, ConnectionPatch
from scipy.signal import convolve2d
import imageProcessing.smoothing as IPSmooth
import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.convolve2D as IPConv2D


# 3×3 Averaging Filter (Initial Smoothing)

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


# 5×5 Gaussian Filter (sigma=1) - Used for smoothing Ixx, Iyy, Ixy

def computeGaussianAveraging(pixel_array, image_width, image_height, kernel_size=5):
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kkernel_size must be an odd number >=3, such as 3, 5, 7, 9")

    # Calculate the appropriate sigma based on the Gaussian kernel
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    kernel_1d = np.exp(-(ax**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()

    averaged = IPConv2D.computeSeparableConvolution2DOddNTapBorderZero(
        pixel_array, image_width, image_height, kernel_1d
    )
    return averaged

# The input image is converted to grayscale and transferred to two-dimensional array directly in memory
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

# Sobel Gradient

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


# Harris Corner Detection: Force return top 1000

def computeHarrisCorners(px_array, image_width, image_height, k=0.04):
    # Compute gradients
    Ix, Iy = computePartialDerivativesSobel(px_array, image_width, image_height)

    # Compute Ixx, Iyy, Ixy
    Ixx = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    Iyy = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    Ixy = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(image_height):
        for c in range(image_width):
            Ixx[r][c] = Ix[r][c] * Ix[r][c]
            Iyy[r][c] = Iy[r][c] * Iy[r][c]
            Ixy[r][c] = Ix[r][c] * Iy[r][c]

    # Apply 5×5 Gaussian smoothing
    Ixx = computeGaussianAveraging(Ixx, image_width, image_height, kernel_size=9)
    Iyy = computeGaussianAveraging(Iyy, image_width, image_height, kernel_size=9)
    Ixy = computeGaussianAveraging(Ixy, image_width, image_height, kernel_size=9)
    # Compute Harris response
    R_vals = IPUtils.createInitializedGreyscalePixelArray(image_width, image_height)
    for r in range(image_height):
        for c in range(image_width):
            detM = Ixx[r][c] * Iyy[r][c] - Ixy[r][c] * Ixy[r][c]
            traceM = Ixx[r][c] + Iyy[r][c]
            R_vals[r][c] = detM - k * (traceM ** 2)

    R_flat = [R_vals[r][c] for r in range(image_height) for c in range(image_width)]
    print(f"[R stats] max={max(R_flat):.2f}, min={min(R_flat):.2f}, avg={sum(R_flat)/len(R_flat):.2f}")

    # Collect candidate corners with threshold=1e7
    threshold = 1e7
    candidates = []
    for r in range(image_height):
        for c in range(image_width):
            if R_vals[r][c] >= threshold:
                candidates.append(((c, r), R_vals[r][c]))

    # If the number is less than the threshold, reserve all pixels as candidates directly (to prevent too few results)
    if len(candidates) < 1000:
        candidates = []
        for r in range(image_height):
            for c in range(image_width):
                candidates.append(((c, r), R_vals[r][c]))

    # Apply 3×3 non-maximum suppression
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

    # Directly return all non-maximum suppressed corners
    print(f"Harris corners found = {len(local_maxima)}")
    final_corners = [pt for (pt, _) in local_maxima]
    return final_corners


def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):
    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width*2, image_height)
    for r in range(image_height):
        for c in range(image_width):
            matchingImage[r][c] = left_pixel_array[r][c]
            matchingImage[r][c + image_width] = right_pixel_array[r][c]
    return matchingImage


# main

def main():
    filename_left_image  = "./images/panoramaStitching/6.1.jpg"
    filename_right_image = "./images/panoramaStitching/6.2.jpg" 

    # filename_left_image  = "./images/panoramaStitching/tongariro_left_01.png"
    # filename_right_image = "./images/panoramaStitching/tongariro_right_01.png"


    (image_width, image_height, px_left_orig) = loadImageasGreyscaleArray(filename_left_image)
    (image_width, image_height, px_right_orig) = loadImageasGreyscaleArray(filename_right_image)

    # 3×3 mean filtering: Initial blur
    px_left_blurred  = computeAveraging3x3(px_left_orig,  image_width, image_height)
    px_right_blurred = computeAveraging3x3(px_right_orig, image_width, image_height)

    # px_left_blurred  = computeGaussianAveraging_sigma1(px_left_orig, image_width, image_height, kernel_size=5)
    # px_right_blurred = computeGaussianAveraging_sigma1(px_right_orig, image_width, image_height, kernel_size=5)

    # Stretch to [0,255], quantify
    px_left  = IPPixelOps.scaleTo0And255AndQuantize(px_left_blurred,  image_width, image_height)
    px_right = IPPixelOps.scaleTo0And255AndQuantize(px_right_blurred, image_width, image_height)

    # Harris Corner detection
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

    fig, axs = pyplot.subplots(1, 2, figsize=(10, 5))
    # Left image + corner point
    axs[0].imshow(px_left, cmap='gray')
    axs[0].set_title("Left with corners")
    for (x, y) in corners_left:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs[0].add_patch(circ)
    # Right picture + corner point
    axs[1].imshow(px_right, cmap='gray')
    axs[1].set_title("Right with corners")
    for (x, y) in corners_right:
        circ = Circle((x, y), 3, color='r', fill=False)
        axs[1].add_patch(circ)
    pyplot.show()


if __name__ == "__main__":
    main()
