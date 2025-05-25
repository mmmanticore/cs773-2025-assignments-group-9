import numpy as np

class CalibrationErrorCalculator():
    def __init__(self) -> None:
        pass
    
    def calculate_error_2D(self, estimate_coordinates, actual_coordinates):
        ####################################################################
        ### You can put all your code for calculating the errors here!!! ###
        ####################################################################
        actual_2D = actual_coordinates[:, 3:5]
        # calculate the errors
        errors = np.linalg.norm(estimate_coordinates - actual_2D, axis=1) # shape: (N,)

        ############################################
        ### Insert all calculated errors here!!! ###
        ############################################
        RMSE = round(np.sqrt(np.mean(errors ** 2)),4)# Root Mean Squared Error (remember to round it to 4 decimal places)
        mean_abs_error_xy = round(np.mean(errors), 4)  # Mean Absolute Error
        std_abs_error_xy = round(np.std(errors), 4) # Standard Deviation of Absolute Error
        abs_errors_xy = errors.tolist()  # A List of Absolute Errors
        std_sqr_error_xy = round(np.std(errors ** 2), 4) # Standard Deviation of Squared Error (remember to round it to 4 decimal places)

        ### DO NOT CHANGE THE RETURN VARIABLES!!!
        return RMSE, mean_abs_error_xy, std_abs_error_xy, abs_errors_xy, std_sqr_error_xy
    
    def calculate_error_3D(self, estimate_coordinates, actual_coordinates):
        ####################################################################
        ### You can put all your code for calculating the errors here!!! ###
        ####################################################################

        est = np.asarray(estimate_coordinates)
        act = np.asarray(actual_coordinates)
        xyz = act[:, :3] if (act.ndim==2 and act.shape[1]>3) else act

        d = np.linalg.norm(est - xyz, axis=1)
        ############################################
        ### Insert all calculated errors here!!! ###
        ############################################
        RMSE       = round(np.sqrt(np.mean(d**2)), 4)# Root Mean Squared Error (remember to round it to 4 decimal places)
        std_sqr_error_xy = round(np.std(d), 4)# Standard Deviation of the Squared Error (remember to round it to 4 decimal places)

        ### DO NOT CHANGE THE RETURN VARIABLES!!!
        return RMSE, std_sqr_error_xy

    
    
    def calculate_stereo_calibration_error(self, left_optical_centre_WRF, estimated_left_3D_coordinates_WRF,
                                           right_optical_centre_WRF, estimated_right_3D_coordinates_WRF, actual_coordinates):
        ####################################################################
        ### You can put all your code for calculating the errors here!!! ###
        ####################################################################

        Lc = np.asarray(left_optical_centre_WRF)
        Rc = np.asarray(right_optical_centre_WRF)
        center_dist = np.linalg.norm(Lc - Rc)

        ############################################
        ### Insert all calculated errors here!!! ###
        ############################################
        Lpts = np.asarray(estimated_left_3D_coordinates_WRF)
        Rpts = np.asarray(estimated_right_3D_coordinates_WRF)
        d = np.linalg.norm(Lpts - Rpts, axis=1)
        RMSE=round(center_dist, 4)  # Root Mean Squared Error (remember to round it to 4 decimal places)
        std_sqr_error_xy = round(np.std(d), 4)# Standard Deviation of the Squared Error (remember to round it to 4 decimal places)

        ### DO NOT CHANGE THE RETURN VARIABLES!!!
        return RMSE, std_sqr_error_xy
