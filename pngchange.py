import os
from PIL import Image
import imageIO.png

def writeRGBPixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    with open(output_filename, 'wb') as outfile:
        writer = imageIO.png.Writer(image_width, image_height, greyscale=False)
        writer.write(outfile, pixel_array)

def convert_image_to_png(input_image_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    try:
        # 打开图片
        with Image.open(input_image_path) as img:
            img = img.convert('RGB')
            width, height = img.size
            pixel_data = list(img.getdata())

            # 转成2D像素数组（RGB扁平展开）
            pixel_array = []
            for y in range(height):
                row = []
                for x in range(width):
                    r, g, b = pixel_data[y * width + x]
                    row.extend([r, g, b])
                pixel_array.append(row)

            # 生成输出路径
            filename = os.path.splitext(os.path.basename(input_image_path))[0]
            output_filename = f"{filename}.png"
            output_path = os.path.join(output_folder, output_filename)

            # 写PNG
            writeRGBPixelArraytoPNG(output_path, pixel_array, width, height)
            print(f"Converted: {input_image_path} -> {output_filename}")

            # # 重新读取保存后的PNG作为Image对象返回
            # converted_img = Image.open(output_path).convert('RGB')
            return output_path  #返回处理完的图片对象

    except Exception as e:
        print(f"Failed to convert {input_image_path}: {e}")
        return None

if __name__ == "__main__":
    input_image_path = "./jpg/2.1.jpg"  # 输入图片路径
    output_folder = "./images/panoramaStitching"  # 输出PNG文件夹
    img_obj = convert_image_to_png(input_image_path, output_folder)

