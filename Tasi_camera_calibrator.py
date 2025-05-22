import numpy as np

class TsaiCameraCalibrator():
    def __init__(self) -> None:
        np.set_printoptions(suppress = True)
    
    def calibrate_2D(self, corner_list, image, camera_type):
        ################################################################
        ### You can put all your code for camera calibration here!!! ###
        ################################################################
        
        
        
        
        #################################################################
        ### Insert all parameters into the Python dictionary below!!! ###
        #################################################################
        required_parameters = {
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0,
            'a4': 0.0,
            'a5': 0.0,
            'a6': 0.0,
            'a7': 0.0,
            'sx': 0.0,
            'tx': 0.0,
            'ty': 0.0,
            'tz': 0.0,
            'f': 0.0,
            'Rt': np.random.rand(4, 4)
        }
        
        # Insert your projected 2D coordinates here!
        projected_2D_coordinates = None
        
        #############################################
        ### DO NOT CHANGE THE RETURN VARIABLES!!! ###
        #############################################
        return projected_2D_coordinates, required_parameters
    
    def back_projection_3D(self, corner_list):
        #############################################################
        ### You can put all your code for back projection here!!! ###
        #############################################################
        
        
        
        
        # Insert your calculated values here!
        estimated_3D_coordinates_WRF = None
        optical_centre_WRF = None
        
        #############################################
        ### DO NOT CHANGE THE RETURN VARIABLES!!! ###
        #############################################
        return estimated_3D_coordinates_WRF, optical_centre_WRF
    