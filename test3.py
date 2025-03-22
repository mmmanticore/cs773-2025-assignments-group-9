import os
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from timeit import default_timer as timer

# 假设你的Harris和Sobel函数都已导入或在本文件中
from test1 import computeHarrisCorners  # 这里换成你自己的Harris函数

# 读取JPG并转灰度pixel_array（内存）
def load_image_as_greyscale_array(image_path):
    img = Image.open(image_path).convert('L')  # 直接转灰度
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

# 可视化 Harris 角点
def visualize_harris(image_array, corners, title):
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    for (x, y) in corners:
        circ = Circle((x, y), 3, color='r', fill=False)
        plt.gca().add_patch(circ)
    plt.show()

def main():
    input_image = './jpg/1.1.jpg'   # 换成你的图片路径
    input_image = './jpg/1.2.jpg'

    # 1️⃣ 直接读取JPG并转灰度
    width, height, pixel_array = load_image_as_greyscale_array(input_image)

    # 2️⃣ Harris角点检测（直接内存跑）
    start = timer()
    harris_corners = computeHarrisCorners(pixel_array, width, height,
                                          k=0.05, threshold_ratio=0.01,
                                          do_nonmax_suppression=True)
    end = timer()
    print(f"Detected {len(harris_corners)} corners in {end - start:.4f} seconds")

    # 3️⃣ 可视化
    visualize_harris(pixel_array, harris_corners, title="Harris Corners")

if __name__ == "__main__":
    main()
