import numpy as np
import cv2

def NearestNeighbourInterpolation(img,kappa):
    def undistPixel(xd,yd,kappa):
        cx=img.shape[1]//2
        cy=img.shape[0]//2
        xu = (xd-cx)*(1+kappa*((xd-cx)**2+(yd-cy)**2))+cx
        yu = (yd-cy)*(1+kappa*((xd-cx)**2+(yd-cy)**2))+cy
        return xu,yu

    #Find known points
    known = dict()
    out = -1*np.ones(img.shape)
    for y in range(0, img.shape[0]*2):
        for x in range(0, img.shape[1]*2):
            x_real = x//2
            y_real = y//2
            xu,yu = undistPixel(x/2,y/2,kappa)
            xu_int = int(round(xu))
            yu_int = int(round(yu))

            #If value is within the image bounds
            if 0<=int(round(xu))<img.shape[1] and 0<=int(round(yu))<img.shape[0]:
                #Find closest matching point
                if (xu_int, yu_int) not in known:
                    known[(xu_int, yu_int)] = (xu_int-xu)**2+(yu_int-yu)**2
                    out[yu_int,xu_int] = img[y_real,x_real]
                else:
                    #If a new point is found closer to the original
                    if known[(xu_int, yu_int)] > (xu_int-xu)**2+(yu_int-yu)**2:
                        known[(xu_int, yu_int)] = (xu_int-xu)**2+(yu_int-yu)**2
                        out[yu_int,xu_int] = img[y_real,x_real]

    out = out.astype(np.uint8)
    return out


def show (img, scale_percent = 30, waitKey=-1):
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
    cv2.imshow('image', img)
    k = cv2.waitKey(waitKey) & 0xFF
    if k == ord('s'):
        cv2.imwrite("image.png", img)
        cv2.destroyAllWindows()     
    if k == ord('q'):
        cv2.destroyAllWindows()  
    cv2.destroyAllWindows()