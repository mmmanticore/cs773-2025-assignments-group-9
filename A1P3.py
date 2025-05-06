import numpy as np
from matplotlib import pyplot as plt
from A1P1 import main as a1p1_main, jisuan_Gaosilvbo
from CS773StitchingSkeleton import prepareMatchingImage
from A1P2 import get_15x15Matrix, Baoli_match

def caculate_singleH(points):
    cp1 = np.array([cp1 for (cp1,cp2) in points])
    cp2 = np.array([cp2 for (cp1,cp2) in points])
    # Take each beginning of the matching point cp1
    n=cp1.shape[0]
    assert n>=4 #At least 4 points are needed to calculate the homography matrix
    A=[]
    for i in range(n):
        x,y = cp1[i]
        u,v= cp2[i]
        A.append([x,y,1,0,0,0,-u*x,-u*y,-u])
        A.append([0,0,0,x,y,1,-v*x,-v*y,-v])
    A=np.array(A)
    #Perform SVD decomposition on A
    U,S,V=np.linalg.svd(A)
    #Take the last column of V
    H=V[-1].reshape(3,3)
    #normalization
    H=H/H[2,2]
    return H

def randomlySelectFourDistinctMatches(matches):
    #Randomly select 4 points
    matches = np.asarray(matches)

    n=matches.shape[0]
    idx=np.random.choice(n,4,replace=False)
    return matches[idx]


def pointsAreCollinear(point1, point2, point3):
    #Determine whether three points are collinear
    #Three points collinear judgment formula area = (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    x1,y1=point1
    x2,y2=point2
    x3,y3=point3
    return (x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))==0


def mappingError(k, transformation):
    #Calculate mapping error
    (x1, y1), (x2, y2) = k
    #Constructing homogeneous coordinate vectors
    vec = np.array([x1, y1, 1.0])
    #Calculate the point (coordinates) after transformation (projection)
    x1h, y1h, w1h = np.dot(transformation, vec)
    #x1h=x1',...
    x1h,y1h=x1h/w1h,y1h/w1h
    #Error in calculating Euclidean distance
    return np.sqrt((x1h-x2)**2+(y1h-y2)**2)

#codes from professor's ppt which we do the futher coding
def ransac_singleH(matches,threshold=5,numberOfRandomDraws=1000):
    bestInlierCount=0
    bestTransformation= None
    for i in range(numberOfRandomDraws):
        seedGroup = randomlySelectFourDistinctMatches(matches)
        # Only take the first two coordinates of each match
        if pointsAreCollinear(seedGroup[0][1],seedGroup[1][1],seedGroup[2][1]):
            continue  # Skip if collinear
        transformation=caculate_singleH(seedGroup)
        #Using threshold to filter inliers
        inliers=[]
        for match in matches:
            map_err=mappingError(match,transformation)
            print(map_err)
            if map_err<0.5:
                inliers.append(match)
        #If there are more inliers, update the optimal
        if len(inliers)>bestInlierCount:
            bestInlierCount=len(inliers)
            bestTransformation=transformation
            bestInliers=inliers
    #Run DLT with all optimal interior points to get finalTransformation
    finalTransformation=caculate_singleH(bestInliers)
    return finalTransformation,bestInliers

def inverse_wrapping(H_final,
                     px_left, px_right,
                     wL, hL, wR, hR):

    # First, project the four corners of the right image to the coordinate system of the left image.
    H_inv = np.linalg.inv(H_final)
    corners_R = np.array([
        [0,   0,   1],
        [wR-1,0,   1],
        [0,   hR-1,1],
        [wR-1,hR-1,1]
    ]).T  # shape=(3,4)
    warped = H_inv @ corners_R           # shape=(3,4)
    warped /= warped[2:3,:]             # Homogeneous Normalization
    xs = warped[0,:]
    ys = warped[1,:]

    # At the same time, consider the range of the left image itself [0,wL)×[0,hL)
    all_x = np.hstack((xs, [0, wL-1]))
    all_y = np.hstack((ys, [0, hL-1]))
    min_x, max_x = np.floor(all_x.min()), np.ceil(all_x.max())
    min_y, max_y = np.floor(all_y.min()), np.ceil(all_y.max())

    # The width and height of the new canvas + the left image placement offset
    new_w = int(max_x - min_x + 1)
    new_h = int(max_y - min_y + 1)
    off_x = int(-min_x)
    off_y = int(-min_y)

    pano = np.zeros((new_h, new_w))

    # Paste the left image to (off_x,off_y)
    pano[off_y:off_y+hL, off_x:off_x+wL] = px_left

    # Reverse sampling: For all pixels on the new canvas, map them to the right image and sample them
    for y in range(new_h):
        for x in range(new_w):
            # Points in the coordinate system of the left figure
            xm = x - off_x
            ym = y - off_y
            p = np.array([xm, ym, 1.0])

            # Use H_final (left to right) to calculate the coordinates of the right image
            p2 = H_final @ p
            x2 = p2[0] / p2[2]
            y2 = p2[1] / p2[2]

            xi = int(np.floor(x2))
            yi = int(np.floor(y2))
            if 0 <= xi < wR and 0 <= yi < hR:
                pano[y, x] = px_right[yi, xi]

    return pano





def main():
    # Pass in corner points, grayscale images and other parameters from A1P1
    corners_left, corners_right, height_left, height_right, width_left, width_right, px_left, px_right = a1p1_main()
    # Calculate the grayscale image after Gaussian filtering (same as A1P2)
    px_left = jisuan_Gaosilvbo(width_left, height_left, px_left)
    px_right = jisuan_Gaosilvbo(width_right, height_right, px_right)
    # Convert to numpy array
    px_left = np.asarray(px_left)
    px_right = np.asarray(px_right)
    # from Phase 2: Generate 15x15 descriptors and brute force match
    des_left = get_15x15Matrix(height_left, width_left, corners_left, px_left)
    des_right = get_15x15Matrix(height_right, width_right, corners_right, px_right)
    matches = Baoli_match(des_left, des_right, ratio=0.9)#According to the return value of a1p2, a triple is returned, including (x1, y1), (x2, y2), score
    #So it needs to be converted into the form of (x1, y1, x2, y2)
    matches = [((x1, y1),(x2, y2)) for (x1, y1), (x2, y2), _ in matches]

    print(f"Phase 3: received {len(matches)} candidate matches from Phase 2.")

    #RANSAC estimates the homography matrix
    H_best, inlier_idxs = ransac_singleH(matches)
    print(f"RANSAC 找到 {len(inlier_idxs)} 个内点。")
    # Extract the inlier coordinate list of the left and right images
    left_inliers = np.array([pt1 for (pt1, pt2) in inlier_idxs])
    right_inliers = np.array([pt2 for (pt1, pt2) in inlier_idxs])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Left picture
    axes[0].imshow(px_left, cmap='gray')
    axes[0].scatter(left_inliers[:, 0], left_inliers[:, 1],
                    s=5, c='r', marker='o')
    axes[0].set_title("Left Image Inliers")
    axes[0].axis('off')

    # The right one
    axes[1].imshow(px_right, cmap='gray')
    axes[1].scatter(right_inliers[:, 0], right_inliers[:, 1],
                    s=5, c='r', marker='o')
    axes[1].set_title("Right Image Inliers")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()    #H_final,=ransac_singleH(matches)
    #Warp and Stitch
    stitched = inverse_wrapping(
        H_best,
        px_left, px_right,
        width_left, height_left,
        width_right, height_right
    )
    # visualize
    plt.figure(figsize=(10,5))
    plt.imshow(stitched, cmap='gray')
    plt.axis('off')
    plt.title("Stitched Panorama (Phase 3)")
    plt.show()

if __name__ == "__main__":
    main()