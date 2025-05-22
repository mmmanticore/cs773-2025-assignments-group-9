import numpy as np

class LineFittingAlgorithm():
    def __init__(self) -> None:
        np.random.seed(689)
        self.distance_threshold = 2
    
    def run(self, corner_groups):
        if corner_groups == None:
            return None, None
        
        best_slope_list = []
        best_intercept_list = []
        for corners in corner_groups:
            corners = np.array(corners, dtype=list)
            number_of_rows = corners.shape[0]
            largest_num_inlier = 0
            best_slope = 0
            best_intercept = 0
            for it_num in range(1000):
                rand_indices = np.random.choice(number_of_rows, size=2, replace=False)
                rand_corners = corners[rand_indices, :]

                x1, y1 = rand_corners[0]
                x2, y2 = rand_corners[1]
                if abs(x2 - x1) <= 5: continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - (slope * x1)

                num_inliers = self.compute_num_inliers(corners, slope, intercept)
                if largest_num_inlier < num_inliers:
                    largest_num_inlier = num_inliers
                    best_slope = slope
                    best_intercept = intercept
            
            best_slope_list.append(best_slope)
            best_intercept_list.append(best_intercept)
        
        return best_slope_list, best_intercept_list


    def compute_num_inliers(self, corners, slope, intercept):
        num_inliers = 0
        for x, y in corners:
            distance = abs(y - (slope * x + intercept))
            if distance <= self.distance_threshold:
                num_inliers += 1
        
        return num_inliers
