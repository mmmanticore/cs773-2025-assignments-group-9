import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch

import SUSAN
import A1P1

from CS773StitchingSkeleton import prepareMatchingImage
import pandas as pd

corners_left, corners_right,height_left,height_right,width_left,width_right, px_left,px_right= SUSAN.main()

px_left=A1P1.jisuan_Gaosilvbo(width_left,height_left,px_left)
px_right=A1P1.jisuan_Gaosilvbo(width_right,height_right,px_right)

def get_15x15Matrix(height, width, corners, pixel,window_size=15):
    des=[]
    half_window_size= window_size // 2
    for ((x,y),R) in corners:
        # 不许超过图片范围
        if x - half_window_size < 0 or x + half_window_size >= width or y-half_window_size < 0 or y+half_window_size >= height :
            continue

        matrix=[]
        for i in range(y-half_window_size,y+half_window_size+1):
            row=[]
            for j in range(x-half_window_size,x+half_window_size+1):
                row.append(pixel[i][j])
            matrix.append(row)

        # 转化为numpy数组
        matrix=np.array(matrix,dtype=np.float32)

        matrix = matrix.flatten()
        # 求均值求方差
        mean = np.mean(matrix)

        # （fi-f~）/sqrt(sum((fi-f~)^2))
        fenmu = np.sqrt(np.sum((matrix - mean) ** 2))
        fenzi = matrix - mean
        descriptor = fenzi / fenmu
        descriptor=descriptor.reshape(matrix.shape)

        des.append({'position':(x,y),'des':descriptor})
    return des


# 暴力匹配
def Baoli_match(list_f,list_g,ratio=0.9):
    final_match=[]
    for f in list_f:
        best_ncc = -1.0
        second_ncc= -1.0
        best_match = None
        for g in list_g:
            # print(list_f[0]['des'])
            ncc=np.dot(f['des'],g['des'])
            if ncc > best_ncc:
                second_ncc =best_ncc
                best_ncc=ncc
                best_match=g
            elif ncc > second_ncc:
                second_ncc=ncc
        if best_ncc>0 and (second_ncc / best_ncc)<ratio:
            final_match.append((f['position'],best_match['position'],best_ncc))
    return final_match


# def test(matrix_f):
#     matrix=np.array(matrix_f,dtype=np.float32)
#     matrix=matrix.flatten()
#     # 求均值求方差
#     mean = np.mean(matrix)
#
#     # （fi-f~）/sqrt(sum((fi-f~)^2))
#     fenmu = np.sqrt(np.sum((matrix-mean)**2))
#     fenzi = matrix - mean
#     descriptor = fenzi / fenmu
#     print(type(descriptor))


def main():
    # 计算描述子
    des_left=get_15x15Matrix(height_left, width_left, corners_left, px_left)
    des_right=get_15x15Matrix(height_right, width_right, corners_right, px_right)

    print(f"Image Left: {len(des_left)} keypoints with descriptors.")
    print(f"Image Right: {len(des_right)} keypoints with descriptors.")

    cp=Baoli_match(des_left,des_right)
    print(f"Found {len(cp)} matches.")
    print(cp[0])


    def save_matches_to_csv(cp, out_file):
        # cp 是 [( (x1,y1), (x2,y2), score ), ... ]
        data = []
        for ((x1, y1), (x2, y2), score) in cp:
            data.append([x1, y1, x2, y2, score])
        df = pd.DataFrame(data, columns=["x1", "y1", "x2", "y2", "score"])
        df.to_csv(out_file, index=False)
    save_matches_to_csv(cp,"matches.csv")

    # 可视化
    matchingImage = prepareMatchingImage(px_left, px_right, width_left, height_left)
    plt.imshow(matchingImage, cmap='gray')
    ax = plt.gca()
    ax.set_title("Matching Image")

    for match in cp:
        left_pt, right_pt, score = match
        x1, y1 = left_pt
        x2, y2 = right_pt
        # 右图坐标偏移 width_left
        x2_shifted = x2 + width_left

        # 在图上找到两个匹配点
        circ_left = plt.Circle((x1, y1), radius=3, fill=False, linewidth=1)
        circ_right = plt.Circle((x2_shifted, y2), radius=3, fill=False, linewidth=1)
        ax.add_artist(circ_left)
        ax.add_artist(circ_right)

        connection = ConnectionPatch(xyA=(x1, y1), xyB=(x2_shifted, y2),
                                     coordsA="data", coordsB="data",
                                     axesA=ax, axesB=ax,
                                     edgecolor='r', linewidth=1)
        ax.add_artist(connection)

    plt.show()


if __name__ == "__main__":
    main()