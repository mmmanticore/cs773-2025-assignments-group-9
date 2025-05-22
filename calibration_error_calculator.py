import numpy as np

class CalibrationErrorCalculator():
    def __init__(self) -> None:
        pass
    
    def calculate_error_2D(self, estimate_coordinates, actual_coordinates):
        ####################################################################
        ### You can put all your code for calculating the errors here!!! ###
        ####################################################################
        
        
        
        
        
        
        
        
        
        
        ############################################
        ### Insert all calculated errors here!!! ###
        ############################################
        RMSE = 0  # Root Mean Squared Error (remember to round it to 4 decimal places)
        mean_abs_error_xy = 0  # Mean Absolute Error
        std_abs_error_xy = 0  # Standard Deviation of Absolute Error
        abs_errors_xy = 0  # A List of Absolute Errors
        std_sqr_error_xy = 0  # Standard Deviation of Squared Error (remember to round it to 4 decimal places)
        
        ### DO NOT CHANGE THE RETURN VARIABLES!!!
        return RMSE, mean_abs_error_xy, std_abs_error_xy, abs_errors_xy, std_sqr_error_xy
    
    
    def calculate_error_3D(self, estimate_coordinates, actual_coordinates):
        ####################################################################
        ### You can put all your code for calculating the errors here!!! ###
        ####################################################################
        
        
        
        
        
        
        
        
        
        
        ############################################
        ### Insert all calculated errors here!!! ###
        ############################################
        RMSE = 0  # Root Mean Squared Error (remember to round it to 4 decimal places)
        std_sqr_error_xy = 0  # Standard Deviation of the Squared Error (remember to round it to 4 decimal places)
        
        ### DO NOT CHANGE THE RETURN VARIABLES!!!
        return RMSE, std_sqr_error_xy
    
    
    def calculate_stereo_calibration_error(self, left_optical_centre_WRF, estimated_left_3D_coordinates_WRF,
                                           right_optical_centre_WRF, estimated_right_3D_coordinates_WRF, actual_coordinates):
        ####################################################################
        ### You can put all your code for calculating the errors here!!! ###
        ####################################################################
        
        
        
        
        
        
        
        
        
        
        ############################################
        ### Insert all calculated errors here!!! ###
        ############################################
        RMSE = 0  # Root Mean Squared Error (remember to round it to 4 decimal places)
        std_sqr_error_xy = 0  # Standard Deviation of the Squared Error (remember to round it to 4 decimal places)
        
        ### DO NOT CHANGE THE RETURN VARIABLES!!!
        return RMSE, std_sqr_error_xy
    