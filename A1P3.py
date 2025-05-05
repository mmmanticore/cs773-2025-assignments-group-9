import numpy as np
from matplotlib import pyplot as plt
from A1P1 import main as a1p1_main, jisuan_Gaosilvbo
from CS773StitchingSkeleton import prepareMatchingImage
from A1P2 import get_15x15Matrix, Baoli_match

def caculate_singleH(points):
    cp1 = np.array([cp1 for (cp1,cp2) in points])
    cp2 = np.array([cp2 for (cp1,cp2) in points])
    #取匹配点cp1的每一个开头
    n=cp1.shape[0]
    assert n>=4 #至少需要4个点来计算单应矩阵
    A=[]
    for i in range(n):
        x,y = cp1[i]
        u,v= cp2[i]
        A.append([x,y,1,0,0,0,-u*x,-u*y,-u])
        A.append([0,0,0,x,y,1,-v*x,-v*y,-v])
    A=np.array(A)
    #对A进行SVD分解
    U,S,V=np.linalg.svd(A)
    #取V的最后一列
    H=V[-1].reshape(3,3)
    #归一化
    H=H/H[2,2]
    return H

def randomlySelectFourDistinctMatches(matches):
    #随机选择4个点
    matches = np.asarray(matches)

    n=matches.shape[0]
    idx=np.random.choice(n,4,replace=False)
    return matches[idx]


def pointsAreCollinear(point1, point2, point3):
    #判断三点是否共线
    #三点共线判断公式area = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    x1,y1=point1
    x2,y2=point2
    x3,y3=point3
    return (x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))==0


def mappingError(k, transformation):
    #计算映射误差
    (x1, y1), (x2, y2) = k
    #构造齐次坐标向量
    vec = np.array([x1, y1, 1.0])
    #计算变换（投影）后的点（坐标）
    x1h, y1h, w1h = np.dot(transformation, vec)
    #x1h=x1',...
    x1h,y1h=x1h/w1h,y1h/w1h
    #计算欧式距离的误差
    return np.sqrt((x1h-x2)**2+(y1h-y2)**2)


def ransac_singleH(matches,threshold=5,numberOfRandomDraws=1000):
    bestInlierCount=0
    bestTransformation= None
    for i in range(numberOfRandomDraws):
        seedGroup = randomlySelectFourDistinctMatches(matches)
        # 只取每个 match 的前两个坐标
        if pointsAreCollinear(seedGroup[0][1],seedGroup[1][1],seedGroup[2][1]):
            continue  # Skip if collinear
        transformation=caculate_singleH(seedGroup)
        #用阈值筛选内点
        inliers=[]
        for match in matches:
            map_err=mappingError(match,transformation)
            print(map_err)
            if map_err<0.5:
                inliers.append(match)
        #如果内点更多，就更新最优
        if len(inliers)>bestInlierCount:
            bestInlierCount=len(inliers)
            bestTransformation=transformation
            bestInliers=inliers
    #用全部最优内点跑DLT,得到finalTransformation
    finalTransformation=caculate_singleH(bestInliers)
    return finalTransformation,bestInliers

def inverse_wrapping(H_final,
                     px_left, px_right,
                     wL, hL, wR, hR):

    # 1) 先把右图四角映到左图坐标系
    H_inv = np.linalg.inv(H_final)
    corners_R = np.array([
        [0,   0,   1],
        [wR-1,0,   1],
        [0,   hR-1,1],
        [wR-1,hR-1,1]
    ]).T  # shape=(3,4)
    warped = H_inv @ corners_R           # shape=(3,4)
    warped /= warped[2:3,:]              # 齐次归一
    xs = warped[0,:]
    ys = warped[1,:]

    # 2) 同时考虑左图本身的范围 [0,wL)×[0,hL)
    all_x = np.hstack((xs, [0, wL-1]))
    all_y = np.hstack((ys, [0, hL-1]))
    min_x, max_x = np.floor(all_x.min()), np.ceil(all_x.max())
    min_y, max_y = np.floor(all_y.min()), np.ceil(all_y.max())

    # 3) 新画布的宽高 + 左图放置偏移
    new_w = int(max_x - min_x + 1)
    new_h = int(max_y - min_y + 1)
    off_x = int(-min_x)
    off_y = int(-min_y)

    pano = np.zeros((new_h, new_w))

    # 4) 把左图贴到 (off_x,off_y)
    pano[off_y:off_y+hL, off_x:off_x+wL] = px_left

    # 5) 逆向采样：对新画布上所有像素，映到右图，采样
    for y in range(new_h):
        for x in range(new_w):
            # 在左图坐标系里的点
            xm = x - off_x
            ym = y - off_y
            p = np.array([xm, ym, 1.0])

            # 用 H_final（左→右）算到右图坐标
            p2 = H_final @ p
            x2 = p2[0] / p2[2]
            y2 = p2[1] / p2[2]

            xi = int(np.floor(x2))
            yi = int(np.floor(y2))
            if 0 <= xi < wR and 0 <= yi < hR:
                pano[y, x] = px_right[yi, xi]

    return pano





def main():
    # —— 从 A1P1 传入角点、灰度图等参数 ——
    corners_left, corners_right, height_left, height_right, width_left, width_right, px_left, px_right = a1p1_main()
    # —— 计算高斯滤波后的灰度图（同 A1P2） ——
    px_left = jisuan_Gaosilvbo(width_left, height_left, px_left)
    px_right = jisuan_Gaosilvbo(width_right, height_right, px_right)
    # **转成 numpy 数组**
    px_left = np.asarray(px_left)
    px_right = np.asarray(px_right)
    # —— Phase 2：生成 15x15 描述子并暴力匹配 ——
    des_left = get_15x15Matrix(height_left, width_left, corners_left, px_left)
    des_right = get_15x15Matrix(height_right, width_right, corners_right, px_right)
    matches = Baoli_match(des_left, des_right, ratio=0.9)#根据a1p2的返回值，返回的是一个三元组，包括了（x1,y1）,(x2,y2),score
    #所以需要转换为（x1,y1,x2,y2）的形式
    matches = [((x1, y1),(x2, y2)) for (x1, y1), (x2, y2), _ in matches]

    print(f"Phase 3: received {len(matches)} candidate matches from Phase 2.")

    # —— RANSAC 估计单应矩阵 ——
    H_best, inlier_idxs = ransac_singleH(matches)
    print(f"RANSAC 找到 {len(inlier_idxs)} 个内点。")

    #H_final,=ransac_singleH(matches)
    # —— Warp and Stitch ——
    stitched = inverse_wrapping(
        H_best,
        px_left, px_right,
        width_left, height_left,
        width_right, height_right
    )
    # —— 可视化拼接结果 ——
    plt.figure(figsize=(10,5))
    plt.imshow(stitched, cmap='gray')
    plt.axis('off')
    plt.title("Stitched Panorama (Phase 3)")
    plt.show()

if __name__ == "__main__":
    main()