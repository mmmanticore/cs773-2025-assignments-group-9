import numpy as np
import math

from line_fitting_algorithm import LineFittingAlgorithm

class DistortionRemover():
    def __init__(self) -> None:
        self.fitter = LineFittingAlgorithm()

    def compute_undistorted_coordinates(self,k1, xd, yd, image_height, image_width):
        image_center_x = image_width / 2
        image_center_y = image_height / 2
        undistorted_corners = []
        # 归一化坐标
        xd = xd - image_center_x
        yd = yd - image_center_y
        # 计算归一化半径
        r_2 = float(xd) ** 2 + float(yd) ** 2
        # print(r_2)
        # 计算归一化畸变因子
        xu = float(xd) * (1 + float(k1) * r_2)
        yu = float(yd) * (1 + float(k1) * r_2)
        xuu = xu + image_center_x
        yuu = yu + image_center_y
        # undistorted_corners.append((xuu,yuu))
        return xuu, yuu

    def remove(self, distorted_corner_groups, height, width):
        '''
        This should be a list of grouped corner points.
        For example:
        undistorted_corner_groups = [
            [[242, 972], [2383, 483], [298, 2394]],  # group 1
            [[483, 734], [293, 781], [982, 129]], # group 2
        ]
        '''
        k1 = 0
        """x_u = (x_d – c_x) * (1 + κ₁ * [ (x_d – c_x)² + (y_d – c_y)² ]) + c_x  
           y_u = (y_d – c_y) * (1 + κ₁ * [ (x_d – c_x)² + (y_d – c_y)² ]) + c_y  
           """
        groups = distorted_corner_groups
        undistorted_corner_groups = []

        for group in groups:
            undistorted_group = []
            for xd, yd in group:
                xui, yui = self.compute_undistorted_coordinates(0, xd, yd, height, width)
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
        k1=0
        for distorted_points, mi, bi in zip(groups,best_slope_list,best_intercept_list):
            # 点到直线 y=m x + b 的垂直距离的平方
            mse = [((mi * x - y + bi) ** 2) / (mi * mi + 1) for (x, y) in distorted_points]
            mse_list.append(np.mean(mse))

        # smallest_MSE = np.mean(mse_list)
        fade_mse=np.mean(mse_list)
        for i in range(200):
            k1=i*1e-9
            undistorted_groups=[]
            for groupss in distorted_corner_groups:
                undistorted=[self.compute_undistorted_coordinates(k1,xd,yd,height,width)for xd,yd in groupss]
                undistorted_groups.append(undistorted)
            #RANSAC拟合直线
            slope_list, intercept_list = line_fitting_alg.run(undistorted_groups)
            #计算每条直线的mse
            mse_each_list=[]
            # for each_group,each_mi,each_bi in zip(distorted_corner_groups,slope_list,intercept_list):
            #     # 原畸变点到拟合直线y=mx+b的垂直距离平方
            #     mse_each = [((each_mi * x - y + each_bi) ** 2) / (each_mi * each_mi + 1) for (x, y) in each_group]
            #     mse_each_list.append(np.mean(mse_each))
            for undist_group, mi, bi in zip(undistorted_groups, slope_list, intercept_list):
                mse_each = [((mi * x - y + bi)**2)/(mi**2+1) for (x,y) in undist_group]
                mse_each_list.append(np.mean(mse_each))


            average=np.mean(mse_each_list)
            #判断最优记录
            if average<fade_mse:
                fade_mse=average
                kappa_1 =k1
                #若阈值小于4,则退出
            if average < 4:
                break


        kappa_2 = None
        average_kappa_value = None

        return fade_mse, kappa_1, kappa_2, average_kappa_value   # DO NOT CHANGE THIS LINE!!!
