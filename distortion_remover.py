import numpy as np
import math

from line_fitting_algorithm import LineFittingAlgorithm

class DistortionRemover():
    def __init__(self) -> None:
        self.fitter = LineFittingAlgorithm()

    def compute_undistorted_coordinates(self,k1, xd, yd, image_width, image_height):
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        undistorted_corners = []
        # normalised coordinate
        xd = xd - image_center_x
        yd = yd - image_center_y
        # Calculate the normalised radius
        r_2 = float(xd) ** 2 + float(yd) ** 2
        # print(r_2)
        # Calculate the normalised distortion factor
        xu = float(xd) * (1 + float(k1) * r_2)
        yu = float(yd) * (1 + float(k1) * r_2)
        xuu = xu + image_center_x
        yuu = yu + image_center_y
        # undistorted_corners.append((xuu,yuu))
        return xuu, yuu

    def remove(self, distorted_corner_groups,width,height):
        '''
        This should be a list of grouped corner points.
        For example:
        undistorted_corner_groups = [
            [[242, 972], [2383, 483], [298, 2394]],  # group 1
            [[483, 734], [293, 781], [982, 129]], # group 2
        ]
        '''
        groups = distorted_corner_groups
        undistorted_corner_groups = []

        for group in groups:
            undistorted_group = []
            for xd, yd in group:
                xui, yui = self.compute_undistorted_coordinates(0, xd, yd, width, height)
                undistorted_group.append([xui, yui])
            undistorted_corner_groups.append(undistorted_group)

        #print("undistorted_corner_groups:", undistorted_corner_groups)

        # A helper function to compute slope and y-intercept for each point in the group
        line_fitting_alg = LineFittingAlgorithm()
        best_slope_list, best_intercept_list = line_fitting_alg.run(undistorted_corner_groups)
        # print("best_slope_list:", best_slope_list, "best_intercept_list:", best_intercept_list)

        # Assign your computed values here!
        # If you choose to not estimate kappa 2, you can leave kappa_2 and average_kappa_value as they are.
        # But DO NOT delete them below!
        mse_list = []
        for distorted_points, mi, bi in zip(groups,best_slope_list,best_intercept_list):
            # The square of the perpendicular distance from the point to the line y = m x + b
            mse = [((mi * x - y + bi) ** 2) / (mi * mi + 1) for (x, y) in distorted_points]
            mse_list.append(np.mean(mse))


        # smallest_MSE = np.mean(mse_list)
        fade_mse=np.mean(mse_list)
        kappa_1 = 0.0
        history = []
        for i in range(200):
            k1 =  i * 1e-9
            undistorted_groups=[]

            for groupss in distorted_corner_groups:
                undistorted=[self.compute_undistorted_coordinates(k1,xd,yd,width,height)for xd,yd in groupss]
                undistorted_groups.append(undistorted)
            #RANSAC fits a straight line
            slope_list, intercept_list = line_fitting_alg.run(undistorted_groups)
            #Calculate the mse for each line
            mse_each_list=[]
            ## Square of the perpendicular distance from the original aberration point to the fitted line y=mx+b
            for origin_group, mi, bi in zip(undistorted_groups, slope_list, intercept_list):
                mse_each = [((mi * x - y + bi)**2)/(mi**2+1) for (x,y) in origin_group]
                mse_each_list.append(np.mean(mse_each))


            average=np.mean(mse_each_list)
            history.append((k1, average))
            #Determining the optimal record
            if average < fade_mse:
                fade_mse, kappa_1 = average, k1
            if fade_mse < 4:
                break


        kappa_2 = None
        average_kappa_value = None

        return fade_mse,kappa_1, kappa_2, average_kappa_value   # DO NOT CHANGE THIS LINE!!!