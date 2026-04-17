import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert, savgol_filter
from scipy.interpolate import interp1d

class SPRPhaseRetrieval:
    def __init__(self):
        self.theta = None
        self.R = None
        self.R_smooth = None
        self.n = None
        self.k = None
        self.phase = None
        
    def load_data_from_csv(self, file_path):
        data = pd.read_csv(file_path)
        self.theta = data["index"].values
        self.R = data["Centroid_step_1"].values
        
    def preprocess_data(self, window_length, polyorder, interpolation_step=0.01):
        # 確保反射率數據是正值
        self.R = np.maximum(self.R, 1e-6)
        
        # 線性插值 - 讓角度數據更細緻
        interp_func = interp1d(self.theta, self.R, kind='linear')
        theta_new = np.arange(self.theta.min(), self.theta.max(), interpolation_step)
        R_interpolated = interp_func(theta_new)
        
        self.theta = theta_new
        self.R = R_interpolated
        
        # 數據平滑 (Savitzky-Golay 濾波器)
        self.R_smooth = savgol_filter(self.R, window_length, polyorder)
        
    def kramers_kronig_transform(self):
        # 使用 Hilbert 變換計算 KK 變換
        analytic_signal = hilbert(self.R_smooth)
        self.n = np.real(analytic_signal)  # 實部 
        self.k = np.imag(analytic_signal)  # 虛部 
        self.phase = np.angle(analytic_signal)  # 相位（單位：弧度）
        
    def find_resonance_angle(self):
        return self.theta[np.argmin(self.R_smooth)]
        
    def plot_results(self):
        resonance_angle = self.find_resonance_angle()
        x_min, x_max = np.min(self.theta), np.max(self.theta)
        
        # 設定分析範圍，聚焦共振角度附近
        resonance_idx = np.argmin(self.R_smooth)
        lf = max(0, resonance_idx - 50)  # 共振角度前 50 個點
        rt = min(len(self.theta), resonance_idx + 50)  # 共振角度後 50 個點
        
        # 展開相位以消除 2π 跳變
        phase_unwrapped = np.unwrap(self.phase[lf:rt])
        # 平滑相位數據
        phase_smoothed = savgol_filter(phase_unwrapped, window_length=11, polyorder=2)
        # 計算相位梯度
        phase_gradient = np.gradient(phase_smoothed, self.theta[lf:rt])
        # 找到梯度最大值及其對應角度
        grad_max_value = np.max(np.abs(phase_gradient))
        grad_max_idx = np.argmax(np.abs(phase_gradient))
        theta_max_grad = self.theta[lf:rt][grad_max_idx]  # 梯度最大值對應的角度
        
        # 繪製反射率圖
        plt.figure(figsize=(8, 5))
        plt.plot(self.theta, self.R, 'b.', label='Raw Data', alpha=0.5)
        plt.plot(self.theta, self.R_smooth, 'g-', label='Smoothed Data')
        plt.axvline(x=resonance_angle, color='red', linestyle='--', 
                    label=f'Resonance Angle (Reflectance): {resonance_angle:.2f}°')
        plt.axvline(x=theta_max_grad, color='orange', linestyle='--', 
                    label=f'Max Gradient Angle (Phase): {theta_max_grad:.2f}°')
        plt.xlabel('Incident Angle (°)')
        plt.ylabel('Reflectance')
        plt.legend()
        plt.grid(True)
        plt.title("Reflectance Plot")
        plt.show()
        
        # 繪製實部（n）
        plt.figure(figsize=(8, 5))
        plt.plot(self.theta, self.n, 'g-', label='Real Part (n)')

        plt.xlabel('Incident Angle (°)')
        plt.ylabel('Real Part of Analytic Signal')
        plt.legend()
        plt.grid(True)
        plt.title("Real Part Plot")
        plt.show()

        # 繪製虛部（k）
        plt.figure(figsize=(8, 5))
        plt.plot(self.theta[lf:rt], self.k[lf:rt], 'b-', label='Imaginary Part (k)')
        plt.xlabel('Incident Angle (°)')
        plt.ylabel('Imaginary Part of Analytic Signal')
        plt.legend()
        plt.grid(True)
        plt.title("Imaginary Part Plot")
        plt.show()
        
        # 繪製相位圖
        plt.figure(figsize=(8, 5))
        plt.plot(self.theta[lf:rt], phase_smoothed, 'm-', label='Smoothed Unwrapped Phase (radians)')
        plt.axvline(x=resonance_angle, color='red', linestyle='--', 
                    label=f'Resonance Angle: {resonance_angle:.2f}°')
        plt.axvline(x=theta_max_grad, color='orange', linestyle='--', 
                    label=f'Max Gradient Angle: {theta_max_grad:.2f}°')
        plt.xlabel('Incident Angle (°)')
        plt.ylabel('Unwrapped Phase (radians)')
        plt.legend()
        plt.grid(True)
        plt.title("Smoothed Unwrapped Phase Plot")
        plt.show()

        # 繪製相位梯度圖
        plt.figure(figsize=(8, 5))
        plt.plot(self.theta[lf:rt], phase_gradient, 'c-', label='Phase Gradient (radians/°)')
        plt.axvline(x=resonance_angle, color='red', linestyle='--', 
                    label=f'Resonance Angle: {resonance_angle:.2f}°')
        plt.axvline(x=theta_max_grad, color='orange', linestyle='--', 
                    label=f'Max Gradient Angle: {theta_max_grad:.2f}°, Value: {grad_max_value:.2f} rad/°')
        plt.xlabel('Incident Angle (°)')
        plt.ylabel('Phase Gradient (radians/°)')
        plt.legend()
        plt.grid(True)
        plt.title("Phase Gradient Plot")
        plt.show()
        
        # 打印結果
        print(f"共振角 (基於反射率): {resonance_angle:.2f}°")
        print(f"相位變化最劇烈的角度: {theta_max_grad:.2f}°")
        print(f"最大梯度值: {grad_max_value:.2f} radians/°")


############################################# 更改 file_path #############################################
file_path = r"C:\Users\USER\Desktop\angle.csv"

################################# 更改 'index'(x軸) 'Centroid_step_1'(y軸) ################################
spr = SPRPhaseRetrieval()
spr.load_data_from_csv(file_path, 'index', 'Centroid_step_1')
spr.preprocess_data(window_length=20, polyorder=3)
spr.kramers_kronig_transform()
##########################################################################################################

# 計算並打印共振角
resonance_angle = spr.find_resonance_angle()
print(f"共振角: {resonance_angle:.2f}°")

# 顯示結果
spr.plot_results()