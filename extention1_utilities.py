import numpy as np
import cv2

def NearestNeighbourInterpolation(img, kappa1, kappa2):
    # 内部闭包不再需要额外传 kappa
    def undistPixel(xd, yd):
        cx = img.shape[1] // 2
        cy = img.shape[0] // 2
        # 中心化坐标
        x = xd - cx
        y = yd - cy
        # 计算 r^2
        r2 = x*x + y*y
        # 同时考虑 k1 r^2 和 k2 r^4
        factor = 1 + kappa1 * r2 + kappa2 * (r2**2)
        # 反向映射到畸变前的坐标
        xu = x * factor + cx
        yu = y * factor + cy
        return xu, yu

    # 建立输出图（全部填 -1 以便后面检测空白）
    out = -1 * np.ones(img.shape, dtype=np.float32)
    known = {}

    H, W = img.shape[:2]
    # 在 0..H,0..W 范围内用细分采样防止“孔洞”
    for y2 in range(0, H*2):
        for x2 in range(0, W*2):
            x_real = x2 // 2
            y_real = y2 // 2
            xu, yu = undistPixel(x2/2, y2/2)
            xi = int(round(xu))
            yi = int(round(yu))

            # 如果映射落在图像内
            if 0 <= xi < W and 0 <= yi < H:
                dist2 = (xi - xu)**2 + (yi - yu)**2
                key = (yi, xi)
                # 取距离最近的样本
                if key not in known or dist2 < known[key]:
                    known[key] = dist2
                    out[yi, xi] = img[y_real, x_real]

    # 填充空白（-1）为 0 并转为 uint8
    out[out < 0] = 0
    return out.astype(np.uint8)


def show(img, scale_percent=30, waitKey=-1):
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    img_small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imshow('image', img_small)
    k = cv2.waitKey(waitKey) & 0xFF
    if k == ord('s'):
        cv2.imwrite("image.png", img_small)
    cv2.destroyAllWindows()
