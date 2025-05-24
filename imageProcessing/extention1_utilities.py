import numpy as np
import cv2

def NearestNeighbourInterpolation(img, kappa1, kappa2):
    # Internal closures no longer require additional passing kappa
    def undistPixel(xd, yd):
        cx = img.shape[1] // 2
        cy = img.shape[0] // 2
        #Centralized coordinates
        x = xd - cx
        y = yd - cy
        # compute r^2
        r2 = x*x + y*y
        # both consider k1 r^2 and k2 r^4
        factor = 1 + kappa1 * r2 + kappa2 * (r2**2)
        # Reverse mapping to the coordinates before distortion
        xu = x * factor + cx
        yu = y * factor + cy
        return xu, yu

    # Create an output graph (fill all with -1 to detect blanks later）
    out = -1 * np.ones(img.shape, dtype=np.float32)
    known = {}

    H, W = img.shape[:2]
    # Use subsampling to prevent "holes" in the 0..H,0..W range
    for y2 in range(0, H*2):
        for x2 in range(0, W*2):
            x_real = x2 // 2
            y_real = y2 // 2
            xu, yu = undistPixel(x2/2, y2/2)
            xi = int(round(xu))
            yi = int(round(yu))

            # If the mapping falls within the image
            if 0 <= xi < W and 0 <= yi < H:
                dist2 = (xi - xu)**2 + (yi - yu)**2
                key = (yi, xi)
                # Take the closest sample
                if key not in known or dist2 < known[key]:
                    known[key] = dist2
                    out[yi, xi] = img[y_real, x_real]

    # Fill blanks (-1) with 0 and convert to uint8
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
