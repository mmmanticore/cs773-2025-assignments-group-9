from matplotlib import pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
from timeit import default_timer as timer
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import imageIO.readwrite as IORW
import imageProcessing.pixelops as IPPixelOps
import imageProcessing.utilities as IPUtils
import imageProcessing.smoothing as IPSmooth
import imageProcessing.extention1_utilities

from distortion_remover_extention1 import DistortionRemover
from calibration_error_calculator import CalibrationErrorCalculator
from Tasi_camera_calibrator import TsaiCameraCalibrator



SHOW_IMAGE = True

# this is a helper function that puts together an RGB image for display in matplotlib, given
# three color channels for r, g, and b, respectively
def prepareRGBImageFromIndividualArrays(r_pixel_array,g_pixel_array,b_pixel_array,image_width,image_height):
    rgbImage = []
    for y in range(image_height):
        row = []
        for x in range(image_width):
            triple = []
            triple.append(r_pixel_array[y][x])
            triple.append(g_pixel_array[y][x])
            triple.append(b_pixel_array[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage


# takes two images (of the same pixel size!) as input and returns a combined image of double the image width
def prepareMatchingImage(left_pixel_array, right_pixel_array, image_width, image_height):

    matchingImage = IPUtils.createInitializedGreyscalePixelArray(image_width * 2, image_height)
    for y in range(image_height):
        for x in range(image_width):
            matchingImage[y][x] = left_pixel_array[y][x]
            matchingImage[y][image_width + x] = right_pixel_array[y][x]

    return matchingImage

def group_corners(corner_list):
    x_axis_center_point = 1630
    left_corner_group_1 = []
    left_corner_group_2 = []
    left_corner_group_3 = []

    right_corner_group_1 = []
    right_corner_group_2 = []
    right_corner_group_3 = []
    # loop through all the corners
    for index, row in corner_list.iterrows():
        x = row['corner location x']
        y = row['corner location y']
        if x < x_axis_center_point:
            # corner on the left-side of the chessboard
            if 250 < y < 367:
                left_corner_group_1.append([x, y])
            elif 368 < y < 478:
                left_corner_group_2.append([x, y])
            elif 479 < y < 645:
                left_corner_group_3.append([x, y])
        else:
            # corner on the right-side of the chessboard
            if 283 < y < 390:
                right_corner_group_1.append([x, y])
            elif 391 < y < 516:
                right_corner_group_2.append([x, y])
            elif 517 < y < 652:
                right_corner_group_3.append([x, y])

    left_group_corners = [left_corner_group_1, left_corner_group_2, left_corner_group_3]
    right_group_corners = [right_corner_group_1, right_corner_group_2, right_corner_group_3]
    group_corners_left_right = left_group_corners + right_group_corners
    print("group_corners_left_right: ", group_corners_left_right, "\n")

    return group_corners_left_right


def run_distortion_removal(camera_name='H3', on_testing=False):
    print("================================================")
    print(f"{camera_name} Camera Distortion Removal Task")
    print("================================================")

    ############################################################
    ### Reading image and performing image pre-processing
    ############################################################
    filename_image = f"./images/{camera_name}.png"
    distorted_color_image = np.asarray(Image.open(filename_image))
    (image_width, image_height, distorted_px_array) = IORW.readRGBImageAndConvertToGreyscalePixelArray(filename_image)
    distorted_px_array = IPSmooth.computeGaussianAveraging3x3(distorted_px_array, image_width, image_height)
    distorted_px_array = IPPixelOps.scaleTo0And255AndQuantize(distorted_px_array, image_width, image_height)



    ############################################################
    ### Loading Corners
    ############################################################
    corners = pd.read_excel('./corner list H3.xlsx')


    ############################################################
    ### Grouping Corners
    ############################################################
    group_corners_left_right = group_corners(corners)



    ############################################################
    ### Some simple visualizations
    ############################################################
    if on_testing == False:
        print("Showing Distorted Image...")
        fig1, axs1 = plt.subplots(1, 1)

        for index, row in corners.iterrows():
            x = row['corner location x']
            y = row['corner location y']
            circle = Circle((x, y), 1.5, color='r')
            axs1.add_patch(circle)

        axs1.set_title('Distorted Image')
        axs1.imshow(distorted_px_array, cmap='gray')

        plt.show()



    ############################################################
    ### Removing Distortion
    ############################################################
    distortion_remover = DistortionRemover()
    #distortion_remover.remove(undistorted_corners)
    print("111:",distortion_remover)
    smallest_MSE, kappa_1, kappa_2, average_kappa_value = distortion_remover.remove(group_corners_left_right,image_width,image_height)
    print(f"Smallest MSE: {smallest_MSE}")
    print(f"kappa 1: {kappa_1}")
    print(f"kappa 2: {kappa_2}")
    print(f"Average kappa value: {average_kappa_value}")
    print("================================================\n\n\n\n")
    # Draw red corners + yellow straight lines on the original grayscale image
    # The set of dedistortion corner points used to fit the line
    slopes, intercepts = distortion_remover.fitter.run(group_corners_left_right)

    # drawing
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(distorted_px_array, cmap='gray')
    ax.set_title("RANSAC Line Fitting (Original Distorted)")
    ax.axis('off')

    # Red: Original distortion corner points
    for _, row in corners.iterrows():
        ax.plot(row['corner location x'],
                row['corner location y'],
                'ro', markersize=2)

    # Solid yellow line: fitted line y = m x + b
    xs = np.array([0, image_width])
    for m, b in zip(slopes, intercepts):
        ys = m*xs + b
        ax.plot(xs, ys, color='yellow', linewidth=1)

    plt.show()
    #New: Compare red dots/blue dots/yellow lines in one picture
    #First use the final kappa_1 to distort and calculate all undistorted_groups
    undistorted_groups = [
        [distortion_remover.compute_undistorted_coordinates(
             kappa_1,kappa_2, xd, yd, image_width, image_height)
         for xd, yd in group]
        for group in group_corners_left_right
    ]
    #Perform a RANSAC fit on these dedistorted points to get slopes, intercepts
    slopes2, intercepts2 = distortion_remover.fitter.run(undistorted_groups)

    # drawing
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(distorted_px_array, cmap='gray')
    ax.set_title(f"Distorted (red) vs Undistorted (blue) Corners\nwith Fitted Lines (yellow), k2={kappa_2:.2e}")
    ax.axis('off')

    #Red: Original distortion corner points
    for _, row in corners.iterrows():
        ax.plot(row['corner location x'],
                row['corner location y'],
                'ro', markersize=2)

    #Blue: Corner points after dedistortion
    for grp in undistorted_groups:
        for xu, yu in grp:
            ax.plot(xu, yu, 'bo', markersize=2)

    # Yellow line: straight line fitted by dedistortion points
    xs = np.array([0, image_width])
    for m, b in zip(slopes2, intercepts2):
        ys = m * xs + b
        ax.plot(xs, ys, color='yellow', linewidth=1)

    plt.show()

    undistorted_color = imageProcessing.extention1_utilities.NearestNeighbourInterpolation(
        distorted_color_image,
        kappa_1,kappa_2
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(undistorted_color, cv2.COLOR_BGR2RGB))
    plt.title(f"Undistorted Image (k1={kappa_1:.2e},k2={kappa_2:.2e}, MSE={smallest_MSE:.2f})")
    plt.axis('off')
    plt.show()

    # ===================== =======================z




def run_camera_calibration(camera_type):
    Tsai_calibration = TsaiCameraCalibrator()
    error_calculator = CalibrationErrorCalculator()

    print("================================================")
    print(f"{camera_type} Camera Calibration Task")
    print("Calibration Left Dataset")
    print("================================================")
    Left_Corner_List = pd.read_excel(f"./images/{camera_type}/Left/{camera_type}_Cube_Left.xlsx", index_col=0, usecols="A:D,G:H").to_numpy()
    Undistorted_H3_Cube_Left_Image = cv2.imread(f"./images/{camera_type}/Left/Undistorted_{camera_type}_Cube_Left.png")
    left_projected_2D_coordinates, _ = Tsai_calibration.calibrate_2D(Left_Corner_List, Undistorted_H3_Cube_Left_Image, camera_type=camera_type)
    error = error_calculator.calculate_error_2D(left_projected_2D_coordinates, Left_Corner_List)
    left_RMSE_2D, left_mean_abs_error_xy, left_std_abs_error_xy, left_abs_errors_xy, left_std_sqr_error_2D = error
    print(f"Calibration error in the left image = {left_RMSE_2D}, STD = {left_std_sqr_error_2D}")

    estimated_left_3D_coordinates_WRF, left_optical_centre_WRF = Tsai_calibration.back_projection_3D(Left_Corner_List)
    left_RMSE_3D, left_std_sqr_error_3D = error_calculator.calculate_error_3D(estimated_left_3D_coordinates_WRF, Left_Corner_List)
    print(f"Calibration error in the left cube = {left_RMSE_3D}, STD = {left_std_sqr_error_3D}")


    print()
    print("================================================")
    print(f"{camera_type} Camera Calibration Task")
    print("Calibration Right Dataset")
    print("================================================")
    Tsai_calibration = TsaiCameraCalibrator()
    Right_Corner_List = pd.read_excel(f"./images/{camera_type}/Right/{camera_type}_Cube_Right.xlsx", index_col=0, usecols="A:D,G:H").to_numpy()
    Undistorted_H3_Cube_Right_Image = cv2.imread(f"./images/{camera_type}/Right/Undistorted_{camera_type}_Cube_Right.png")
    right_projected_2D_coordinates, _ = Tsai_calibration.calibrate_2D(Right_Corner_List, Undistorted_H3_Cube_Right_Image, camera_type=camera_type)
    error = error_calculator.calculate_error_2D(right_projected_2D_coordinates, Right_Corner_List)
    right_RMSE_2D, right_mean_abs_error_xy, right_std_abs_error_xy, right_abs_errors_xy, right_std_sqr_error_2D = error
    print(f"Calibration error in the right image = {right_RMSE_2D}, STD = {right_std_sqr_error_2D}")

    estimated_right_3D_coordinates_WRF, right_optical_centre_WRF = Tsai_calibration.back_projection_3D(Right_Corner_List)
    right_RMSE_3D, right_std_sqr_error_3D = error_calculator.calculate_error_3D(estimated_right_3D_coordinates_WRF, Right_Corner_List)
    print(f"Calibration error in the right cube = {right_RMSE_3D}, STD = {right_std_sqr_error_3D}")


    print()
    print("================================================")
    print(f"{camera_type} Camera Calibration Task")
    stereo_cali_error, stero_cali_error_std = error_calculator.calculate_stereo_calibration_error(
        left_optical_centre_WRF,
        estimated_left_3D_coordinates_WRF,
        right_optical_centre_WRF,
        estimated_right_3D_coordinates_WRF,
        Left_Corner_List
    )

    print(f"Stereo calibration error = {stereo_cali_error}, STD = {stero_cali_error_std}")
    print("================================================")




def main():
    run_distortion_removal('H3')

    run_camera_calibration('H3')
    run_camera_calibration('W3')


if __name__ == '__main__':
    main()
