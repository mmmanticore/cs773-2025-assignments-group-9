import numpy as np
import math
import CS773A2Ph1
from line_fitting_algorithm import LineFittingAlgorithm

class DistortionRemover():
    def __init__(self) -> None:
        self.fitter = LineFittingAlgorithm()

    def compute_undistorted_coordinates(self,k1,k2, xd, yd,image_width,image_height):
        # Normalized coordinates
        image_center_x = image_width / 2
        image_center_y = image_height / 2

        xd = xd - image_center_x
        yd = yd - image_center_y
        # Calculate the normalized radius
        r_2 = float(xd) ** 2 + float(yd) ** 2
        special=1+k1*r_2+k2*(r_2**2)
        # Calculate the normalized distortion factor
        #xu = xd(1+κ1ρ^2 + κ2ρ^4 + ...),
        #yu = yd(1 κ1p^2 + κ2ρ^4 + . . .)
        xu = float(xd) *special+image_center_x
        yu = float(yd) * special+image_center_y
        return xu, yu

    def remove(self, distorted_corner_groups,width,height):
        '''
        This should be a list of grouped corner points.
        For example:
        undistorted_corner_groups = [
            [[242, 972], [2383, 483], [298, 2394]],  # group 1
            [[483, 734], [293, 781], [982, 129]], # group 2
        ]
        '''
        """x_u = (x_d – c_x) * (1 + κ₁ * [ (x_d – c_x)² + (y_d – c_y)² ]) + c_x  
           y_u = (y_d – c_y) * (1 + κ₁ * [ (x_d – c_x)² + (y_d – c_y)² ]) + c_y  
           """
        groups = distorted_corner_groups
        undistorted_corner_groups = []



        for group in groups:
            undistorted_group = []
            for xd, yd in group:
                xui, yui = self.compute_undistorted_coordinates(0,0,xd, yd, width, height)
                undistorted_group.append([xui, yui])
            undistorted_corner_groups.append(undistorted_group)

        print("undistorted_corner_groups:", undistorted_corner_groups)

        # A helper function to compute slope and y-intercept for each point in the group
        line_fitting_alg = LineFittingAlgorithm()
        #mi and bi are the slope and intercept of the fitted line for each group
        best_slope_list, best_intercept_list = line_fitting_alg.run(undistorted_corner_groups)
        # print("best_slope_list:", best_slope_list, "best_intercept_list:", best_intercept_list)

        # Assign your computed values here!
        # If you choose to not estimate kappa 2, you can leave kappa_2 and average_kappa_value as they are.
        # But DO NOT delete them below!
        mse_list = []
        for distorted_points, mi, bi in zip(groups,best_slope_list,best_intercept_list):
            # The square of the perpendicular distance from the point to the line y=m x + b
            mse = [((mi * x - y + bi) ** 2) / (mi * mi + 1) for (x, y) in distorted_points]
            mse_list.append(np.mean(mse))

        # smallest_MSE = np.mean(mse_list)
        fade_mse=np.mean(mse_list)
        kappa_1 = 0.0
        history = []
        #Search k1 first
        for i in range(200):
            k1 =  i * 1e-9
            undistorted_groups=[]

            for groupss in distorted_corner_groups:
                undistorted=[self.compute_undistorted_coordinates(k1,0,xd,yd,width,height)for xd,yd in groupss]
                undistorted_groups.append(undistorted)
            #RANSAC fit line
            slope_list, intercept_list = line_fitting_alg.run(undistorted_groups)
            #Calculate the mse of each line
            mse_each_list=[]
             # The square of the vertical distance from the original distortion point to the fitting line y=mx+b
            for origin_group, mi, bi in zip(undistorted_groups, slope_list, intercept_list):
                mse_each = [((mi * x - y + bi)**2)/(mi**2+1) for (x,y) in origin_group]
                mse_each_list.append(np.mean(mse_each))


            average=np.mean(mse_each_list)
            history.append((k1, average))
            #Determine the best record
            if average < fade_mse:
                fade_mse, kappa_1 = average, k1
            if fade_mse < 4:
                break
        best_mse2 = fade_mse
        best_k2 = 0.0
        #Step 2: Fix k1 and optimize k2
        for k2 in np.linspace(-1e-15, 1e-15, 101):
            undist2 = [
                [self.compute_undistorted_coordinates(kappa_1, k2, xd, yd, width, height)
                 for xd, yd in grp]
                for grp in groups
            ]
            slopes2, inters2 = self.fitter.run(undist2)
            mses2 = []
            for ug, m, b in zip(undist2, slopes2, inters2):
                errs = [((m * x - y + b) ** 2) / (m * m + 1) for x, y in ug]
                mses2.append(np.mean(errs))
            mse2 = np.mean(mses2)
            if mse2 < best_mse2:
                best_mse2, best_k2 = mse2, k2

        kappa_2 = best_k2
        final_groups = [
            [self.compute_undistorted_coordinates(kappa_1, best_k2, xd, yd, width, height)
             for xd, yd in grp]
            for grp in groups
        ]
        slopes_f, inters_f = self.fitter.run(final_groups)
        mses_f = []
        for ug, m, b in zip(final_groups, slopes_f, inters_f):
            errs = [((m * x - y + b) ** 2) / (m * m + 1) for x, y in ug]
            mses_f.append(np.mean(errs))
        final_mse = np.mean(mses_f)
        fade_mse = final_mse
        average_kappa_value = None
        return fade_mse,kappa_1, kappa_2, average_kappa_value   # DO NOT CHANGE THIS LINE!!!
