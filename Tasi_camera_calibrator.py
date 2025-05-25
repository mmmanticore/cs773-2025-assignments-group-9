import numpy as np

class TsaiCameraCalibrator():
    def __init__(self) -> None:
        np.set_printoptions(suppress = True)
        self._last_params = None

    
    def calibrate_2D(self, corner_list, image, camera_type):
        ################################################################
        ### You can put all your code for camera calibration here!!! ###
        ################################################################

        world_pts = corner_list[:, :3]    # X, Y, Z
        uv_prime  = corner_list[:, 3:5]   # u', v'
        us = uv_prime[:, 0]
        vs = uv_prime[:, 1]

        # 2. 获取图像尺寸与像素参数
        H, W = image.shape[:2]
        if camera_type == 'H3':
            dx = dy = 0.00155
        elif camera_type == 'W3':
            dx, dy = 0.001691, 0.001663
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")
        cx, cy = W / 2.0, H / 2.0
        sx = 1.0
        print(f"[Step2] dx={dx}, dy={dy}, cx={cx}, cy={cy}, sx={sx}")


        # 3. 感应面归一化坐标
        x_u = sx * dx * (us - cx)
        y_u =      dy * (cy - vs)


        # 4. 构造线性系统 M a = b
        n = len(world_pts)
        M = np.zeros((n, 7))
        b = x_u.copy()
        for i, (X, Y, Z) in enumerate(world_pts):
            M[i] = [ y_u[i]*X, y_u[i]*Y, y_u[i]*Z, y_u[i],
                    -x_u[i]*X, -x_u[i]*Y, -x_u[i]*Z ]
        L = np.linalg.lstsq(M, b, rcond=None)[0]
        a1, a2, a3, a4, a5, a6, a7 = L
        print(f"[Step4] a1..a7= {a1:.6f},{a2:.6f},{a3:.6f},{a4:.6f},{a5:.6f},{a6:.6f},{a7:.6f}")

        # 5. 计算 t_y 和 s_x
        t_y = 1.0 / np.sqrt(a5*a5 + a6*a6 + a7*a7)
        s_x = abs(t_y) * np.linalg.norm([a1, a2, a3])


        # 6. 构造并归一化旋转
        r1 = np.array([a1, a2, a3]) * t_y
        r2 = np.array([a5, a6, a7]) * t_y
        r1 /= s_x
        r2 /= s_x
        r3 = np.cross(r1, r2)
        # R = np.vstack((r1, r2, r3))

        # 7. 最小二乘解 f 和 t_z
        A_ftz = []
        b_ftz = []
        for i, (X, Y, Z) in enumerate(world_pts):
            num = r2.dot([X, Y, Z]) + t_y
            den = r3.dot([X, Y, Z])
            A_ftz.append([num, -y_u[i]])
            b_ftz.append(y_u[i] * den)
        A_ftz = np.vstack(A_ftz)
        b_ftz = np.array(b_ftz)
        f, t_z = np.linalg.lstsq(A_ftz, b_ftz, rcond=None)[0]

        # 8. 单独最小二乘解 t_x
        A_tx = []
        b_tx = []
        for i, (X, Y, Z) in enumerate(world_pts):
            num = r1.dot([X, Y, Z])
            den = r3.dot([X, Y, Z])
            A_tx.append([1.0])
            b_tx.append(x_u[i] * (den + t_z) / f - num)
        t_x = float(np.linalg.lstsq(np.vstack(A_tx), np.array(b_tx), rcond=None)[0])

        # —— Step5：判定 t_y 的符号 —— 
        # us,vs, x_u,y_u, r1,r2,R, t_x,t_y, world_pts 都已算好
        if f < 0:
            f = -f
        idx = np.argmax((us - cx)**2 + (vs - cy)**2)
        X0, Y0, Z0 = world_pts[idx]
        x_new = r1.dot([X0, Y0, Z0]) + t_x
        y_new = r2.dot([X0, Y0, Z0]) + t_y

        if np.sign(x_new) != np.sign(x_u[idx]) or np.sign(y_new) != np.sign(y_u[idx]):
            # 翻转 t_y 和 R 的行向量
            t_y = -t_y
            t_x = -t_x
            r1  = -r1
            r2  = -r2
            r3 = np.cross(r1, r2)
            R  = np.vstack((r1, r2, r3))

        # 9. 构造内参矩阵 K
        K = np.array([
            [sx * f / dx, 0.0,        cx],
            [0.0,         -f / dy,     cy],
            [0.0,         0.0,        1.0]
        ])
        print("[Step9] K=", K)

        # 10. 投影 3D->2D
        projected = []
        for (X, Y, Z) in world_pts:
            cam = R.dot([X, Y, Z]) + np.array([t_x, t_y, t_z])
            u_proj = (K[0,0] * cam[0] + K[0,2] * cam[2]) / cam[2]
            v_proj = (K[1,1] * cam[1] + K[1,2] * cam[2]) / cam[2]
            projected.append([u_proj, v_proj])
        p = world_pts[idx]   # 用同一个 idx

        projected_2D_coordinates = np.array(projected)

        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[0, 3] = t_x
        Rt[1, 3] = t_y
        Rt[2, 3] = t_z

        # #################################################################
        # ### Insert all parameters into the Python dictionary below!!! ###
        # #################################################################

        required_parameters = {
            'a1': a1,
            'a2': a2,
            'a3': a3,
            'a4': a4,
            'a5': a5,
            'a6': a6,
            'a7': a7,
            'sx': s_x,
            'tx': t_x,
            'ty': t_y,
            'tz': t_z,
            'f': f,
            'Rt': Rt,
            'dx':dx,'dy':dy,'cx':cx,'cy':cy,
            'K':K,
            't':np.array([t_x, t_y, t_z]),
        }

        # 11. 打包返回
        # required_parameters = {
        #     'a1':a1,'a2':a2,'a3':a3,'a4':a4,
        #     'a5':a5,'a6':a6,'a7':a7,
        #     'f':f, 'R':R, 't':np.array([t_x, t_y, t_z]),
        #     'Rt': Rt,
        #     'K':K, 'dx':dx,'dy':dy,'cx':cx,'cy':cy,
        #     'sx': s_x, 'tx': t_x, 'ty': t_y, 'tz': t_z,
        # }
        self._last_params = required_parameters
        self._last_projected = projected_2D_coordinates
        return projected_2D_coordinates, required_parameters
        
        

        # required_parameters = {
        #     'a1': 0.0,
        #     'a2': 0.0,
        #     'a3': 0.0,
        #     'a4': 0.0,
        #     'a5': 0.0,
        #     'a6': 0.0,
        #     'a7': 0.0,
        #     'sx': 0.0,
        #     'tx': 0.0,
        #     'ty': 0.0,
        #     'tz': 0.0,
        #     'f': 0.0,
        #     'Rt': np.random.rand(4, 4)
        # }
        
        # # Insert your projected 2D coordinates here!
        # projected_2D_coordinates = None
        
        # #############################################
        # ### DO NOT CHANGE THE RETURN VARIABLES!!! ###
        # #############################################
        # return projected_2D_coordinates, required_parameters
    
    def back_projection_3D(self, corner_list):
        #############################################################
        ### You can put all your code for back projection here!!! ###
        #############################################################
        """
        Use last Rt and intrinsics to back-project 2D->3D given known Z.
        Returns estimated 3D coords (n,3) and optical center (3,)
        """
        if self._last_params is None:
            raise RuntimeError("Run calibrate_2D first.")
        p = self._last_params
        R = p['Rt'][:3,:3]; t = p['Rt'][:3,3]
        K = p['K']    # [[f_x,0,cx],[0,f_y,cy],[0,0,1]]
        P = K @ np.hstack([R, t.reshape(3,1)])
        Xw_est = []
        for row in corner_list:
            if row.shape[0] >= 5:
                X_i, Y_i, Z_i, u_i, v_i = row[:5]
            else:
                X_i, Y_i, u_i, v_i = row; Z_i = 0.0
            p11,p12,p13,p14 = P[0]
            p21,p22,p23,p24 = P[1]
            p31,p32,p33,p34 = P[2]
            A = np.array([
                [p11 - u_i*p31, p12 - u_i*p32],
                [p21 - v_i*p31, p22 - v_i*p32]
            ])
            b = np.array([
                u_i*(p33*Z_i + p34) - (p13*Z_i + p14),
                v_i*(p33*Z_i + p34) - (p23*Z_i + p24)
            ])
            xy = np.linalg.lstsq(A, b, rcond=None)[0]
            Xw_est.append([xy[0], xy[1], Z_i])
        optical_center = -R.T @ t
        return np.array(Xw_est), optical_center
        
        
        # # Insert your calculated values here!
        # estimated_3D_coordinates_WRF = None
        # optical_centre_WRF = None
        
        # #############################################
        # ### DO NOT CHANGE THE RETURN VARIABLES!!! ###
        # #############################################
        # return estimated_3D_coordinates_WRF, optical_centre_WRF
    