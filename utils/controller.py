import vgamepad as vg
import time

class VirtualController:
    def __init__(self):
        """
        初始化虚拟 Xbox 360 手柄
        注意：必须已安装 ViGEmBus 驱动
        """
        try:
            self.gamepad = vg.VX360Gamepad()
            print("Virtual Controller Initialized Successfully.")
        except Exception as e:
            print(f"Failed to initialize gamepad: {e}")
            print("Please ensure ViGEmBus driver is installed.")
            raise e

    def reset_state(self):
        """释放所有按键并重置摇杆"""
        self.gamepad.reset()
        self.gamepad.update()

    def tap_button(self, button, duration=0.1):
        """
        点击按钮 (按下 -> 等待 -> 释放)
        :param button: vgamepad 按钮常量 (如 vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        :param duration: 按住持续时间 (秒)
        """
        self.gamepad.press_button(button)
        self.gamepad.update()
        time.sleep(duration)
        self.gamepad.release_button(button)
        self.gamepad.update()

    def hold_button(self, button):
        """按下并保持按钮 (用于蓄力等)"""
        self.gamepad.press_button(button)
        self.gamepad.update()

    def release_button(self, button):
        """释放按钮"""
        self.gamepad.release_button(button)
        self.gamepad.update()

    def set_left_stick(self, x, y):
        """
        控制左摇杆 (移动)
        :param x: -1.0 (左) 到 1.0 (右)
        :param y: -1.0 (下) 到 1.0 (上)
        """
        self.gamepad.left_joystick_float(x_value_float=x, y_value_float=y)
        self.gamepad.update()

    def set_right_stick(self, x, y):
        """控制右摇杆 (视角)"""
        self.gamepad.right_joystick_float(x_value_float=x, y_value_float=y)
        self.gamepad.update()

    def execute_action(self, action_dict):
        """
        执行复杂的动作指令 (支持组合键、长按、扳机)
        :param action_dict: 包含 buttons, triggers, duration 的字典
        """
        # 0. 设置摇杆 (Stick) - 新增支持
        stick_left = action_dict.get("stick_left", None)
        if stick_left:
            self.gamepad.left_joystick_float(x_value_float=stick_left[0], y_value_float=stick_left[1])

        # 1. 按下所有按键 (Buttons)
        buttons = action_dict.get("buttons", [])
        for btn in buttons:
            self.gamepad.press_button(btn)
            
        # 2. 设置扳机 (Triggers) - L2/R2
        triggers = action_dict.get("triggers", {})
        if "left" in triggers:
            self.gamepad.left_trigger_float(triggers["left"])
        if "right" in triggers:
            self.gamepad.right_trigger_float(triggers["right"])
            
        self.gamepad.update()
        
        # 3. 持续时间 (Hold Duration)
        duration = action_dict.get("duration", 0.1)
        time.sleep(duration)
        
        # 4. 检查是否需要释放按键 (默认为 True)
        # 如果设置为 False，则保持按键按下状态 (用于输入缓冲或长按衔接)
        should_release = action_dict.get("release", True)
        
        if should_release:
            # 释放所有按键
            for btn in buttons:
                self.gamepad.release_button(btn)
                
            # 重置扳机
            if "left" in triggers:
                self.gamepad.left_trigger_float(0.0)
            if "right" in triggers:
                self.gamepad.right_trigger_float(0.0)
                
            # 重置摇杆
            if stick_left:
                self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            
        self.gamepad.update()

if __name__ == "__main__":
    # 简单测试脚本
    print("初始化虚拟手柄...")
    try:
        ctl = VirtualController()
        print("✅ 手柄初始化成功！")
        
        print("请在 3 秒内切换到文本编辑器或游戏窗口...")
        time.sleep(3)
        
        print("发送 'A' 键信号...")
        ctl.tap_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        
        print("测试结束。如果刚才有反应（如输入了空格或确认），说明驱动正常。")
    except Exception as e:
        print(f"❌ 手柄初始化失败: {e}")