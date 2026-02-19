import mss
import numpy as np
import cv2
import time

try:
    import pygetwindow as gw
except ImportError:
    gw = None

class WindowCapture:
    def __init__(self, window_name="Monster Hunter Wilds"):
        """
        初始化屏幕捕获类
        :param window_name: 游戏窗口的标题
        """
        self.window_name = window_name
        self.sct = mss.mss()
        self.monitor = self._get_window_geometry()
        
    def _get_window_geometry(self):
        """
        获取指定窗口的坐标和尺寸
        """
        try:
            if gw is None:
                raise ImportError("pygetwindow module not found. Please install it via 'pip install pygetwindow'")

            # 尝试寻找窗口
            windows = gw.getWindowsWithTitle(self.window_name)
            if not windows:
                raise IndexError(f"Window '{self.window_name}' not found.")

            window = windows[0]
            
            # 定义捕获区域 (left, top, width, height)
            # 注意：某些游戏可能有边框，这里可能需要微调偏移量
            return {
                "top": window.top,
                "left": window.left,
                "width": window.width,
                "height": window.height
            }
        except Exception as e:
            print(f"Warning: Could not find window '{self.window_name}'. Error: {e}")
            print("Defaulting to primary monitor.")
            return self.sct.monitors[1]

    def get_screenshot(self):
        """
        捕获当前帧并转换为 OpenCV 格式 (BGR)
        :return: numpy array image
        """
        # 捕获图像
        screenshot = self.sct.grab(self.monitor)
        
        # 转换为 numpy 数组
        img = np.array(screenshot)
        
        # mss 返回的是 BGRA，我们需要去掉 Alpha 通道并保持 BGR (OpenCV 默认格式)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img

# 调试代码
if __name__ == "__main__":
    # 确保游戏正在运行，或者修改 window_name 为当前打开的某个窗口进行测试
    cap = WindowCapture(window_name="Notepad")
    while True:
        frame = cap.get_screenshot()
        cv2.imshow("Screen Capture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()