import pandas as pd
import cv2
import matplotlib.pyplot as plt

# 1) 读取Excel：E,F列是Raw (u,v)，G,H列是Corrected (u',v')
excel_path = "./images/H3/left/H3_Cube_Left.xlsx"
raw = pd.read_excel(excel_path, index_col=0, usecols="A:D,E:F").to_numpy()
corr = pd.read_excel(excel_path, index_col=0, usecols="A:D,G:H").to_numpy()

u_raw = raw[:, 3]
v_raw = raw[:, 4]
u_corr = corr[:, 3]
v_corr = corr[:, 4]

# 2) 读取去畸变图
undist_img = cv2.imread("./images/H3/left/Undistorted_H3_Cube_Left.png")
undist_rgb = cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB)

# 3) 在同一幅图上画出
plt.figure(figsize=(6,6))
plt.imshow(undist_rgb)            # Matplotlib 默认 origin='upper'
plt.scatter(u_raw, v_raw,
            c='red', s=25, label="Raw (E,F)", alpha=0.7)
plt.scatter(u_corr, v_corr,
            c='lime', s=25, label="Corrected (G,H)", alpha=0.7)
plt.title("Raw vs Corrected Corners on Undistorted Image")
plt.legend(loc='upper right')
plt.axis('off')
plt.show()
