import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import keyboard
import vgamepad as vg

from utils.capture import WindowCapture
from utils.controller import VirtualController
from utils.vision import ImageProcessor
from envs.actions import ACTION_MAP, MOVE_ACTIONS

class MHWildsEnv(gym.Env):
    """
    自定义 Gym 环境用于怪物猎人荒野
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, window_name="Monster Hunter Wilds", render_mode='human'):
        super(MHWildsEnv, self).__init__()
        
        self.window_name = window_name
        self.render_mode = render_mode
        
        # 1. 初始化核心组件
        # 注意：如果找不到窗口，WindowCapture 会默认抓取主屏幕，方便调试
        self.cap = WindowCapture(window_name)
        self.controller = VirtualController()
        self.processor = ImageProcessor(input_shape=(84, 84),
                                       crnn_model_path="models/crnn/damage_crnn_best.pth")
        
        # 2. 定义动作空间
        # 使用 MultiDiscrete 允许同时控制按键和移动
        # 格式: [按键动作ID, 移动动作ID]
        # 例如: [1, 1] 代表 "按下Y键" 且 "向前移动"
        self.action_space = spaces.MultiDiscrete([len(ACTION_MAP), len(MOVE_ACTIONS)])
        
        # 3. 定义观察空间
        # 形状: (H, W, C) -> (84, 84, 1) 灰度图
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(84, 84, 1), 
            dtype=np.uint8
        )
        
        # 状态变量
        self.current_frame = None
        self.last_health = 1.0
        # [修改] 这是一个大范围区域，覆盖屏幕中心，用于捕捉跳出的伤害数字
        # 请使用 tools/calibrate_roi.py 的模式 3 重新校准
        # [用户校准] 覆盖屏幕右侧大部分区域
        self.damage_roi = (924, 25, 1534, 1398)
        self.steps_since_last_hit = 0 # [新增] 记录距离上次命中的步数
        self.damage_baseline = 0.0   # [尖峰检测] 伤害像素的指数移动平均基线
        
        # [新增] 斩位、开刃等级、气刃槽 ROI
        # 默认值仅供参考
        # 1. 斩位 (Sharpness): 武器锋利度图标 (小刀) - 需要重新校准
        self.sharpness_roi = (136, 159, 285, 22) 
        # 2. 气刃槽 (Spirit Gauge): 内部填充量 & 颜色
        self.spirit_gauge_roi = (139, 227, 228, 8)
        self.red_spirit_timer = 0 # [新增] 红刃状态缓冲计时器

        # 暂停控制
        self.paused = False
        # 监听 'P' 键，触发回调
        keyboard.on_press_key('p', self._toggle_pause)
        print("【系统】按 'P' 键可暂停/继续训练。")

    def _toggle_pause(self, event):
        self.paused = not self.paused
        print(f"\n>>> 训练已 {'暂停' if self.paused else '继续'} <<<")

    def step(self, action):
        # --- 0. 暂停逻辑 ---
        while self.paused:
            time.sleep(0.5)
            # 保持渲染，防止窗口无响应，同时允许用户观察当前画面
            self.render()
            # 在暂停期间不执行任何动作，也不返回数据
            continue

        # action 是一个数组 [button_id, move_id]
        button_id, move_id = action
        
        # 初始化奖励
        reward = 0.0
        
        # --- 1. 执行动作 ---
        # 处理移动
        move_vec = MOVE_ACTIONS[move_id]
        self.controller.set_left_stick(move_vec[0], move_vec[1])
        
        # 处理按键
        # 获取动作定义字典
        action_entry = ACTION_MAP[button_id]
        
        # 统一转换为列表处理 (支持单个动作和宏动作)
        if isinstance(action_entry, dict):
            action_sequence = [action_entry]
        else:
            action_sequence = action_entry

        # [新增] 标记本回合是否包含攻击动作 (用于触发伤害窗口)
        is_attacking_step = False

        # 遍历执行序列
        for i, action_def in enumerate(action_sequence):
            buttons = action_def.get("buttons", [])
            triggers = action_def.get("triggers", {})
            
            # 判断当前子动作是否为攻击
            is_sub_attack = (vg.XUSB_BUTTON.XUSB_GAMEPAD_Y in buttons) or \
                            (vg.XUSB_BUTTON.XUSB_GAMEPAD_B in buttons) or \
                            ("right" in triggers) or \
                            ("left" in triggers and vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER in buttons)

            if is_sub_attack:
                is_attacking_step = True

            # [人工干预] 攻击自动对焦 (只在序列的第一个动作触发)
            if i == 0:
                if is_sub_attack:
                    # 1. 先重置视角 (L1/LB) - 确保相机对准怪物 (前提是已锁定)
                    self.controller.tap_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER, duration=0.05)
                    
                    # 2. [新增] 激活集中模式 (L2/LT) - 强迫人物转向相机朝向
                    # 怪物猎人荒野中，L2 集中模式可以让角色迅速调整朝向
                    self.controller.gamepad.left_trigger_float(1.0)
                    self.controller.gamepad.update()
                    time.sleep(0.1) # 稍作等待让人物转身
                    self.controller.gamepad.left_trigger_float(0.0)
                    self.controller.gamepad.update()

            # 执行具体动作
            self.controller.execute_action(action_def)
            
            # 处理后摇 (Post Delay) - 等待动画播放
            delay = action_def.get("post_delay", 0.0)
            if delay > 0:
                time.sleep(delay)
            
        # 如果本回合执行了攻击，开启伤害检测窗口 (松开按键后)
        # 窗口持续约 0.8 秒 (8帧)，覆盖伤害数字显现和停留的时间
        if is_attacking_step:
            self.damage_window = 8 

        # --- [修复时序] 立即截取新帧用于伤害检测 ---
        # 旧代码在此处使用上一步的 self.current_frame，导致奖励归因到错误的动作
        self.current_frame = self.cap.get_screenshot()

        # --- 伤害奖励逻辑 (轮廓形状过滤 + 尖峰检测) ---
        DAMAGE_REWARD_SCALE = 50.0  # 50点伤害 ≈ 1.0 奖励 (可调)
        
        current_damage_sum = 0
        hit_count = 0

        # [优化] 仅在伤害窗口开启时进行检测，避免闲逛时的误判
        if self.damage_window > 0:
            self.damage_window -= 1
            current_damage_sum, hit_count, _, _ = self.processor.detect_hit_signals(
                self.current_frame, self.damage_roi
            )

        # 尖峰检测：只有当像素面积显著超过基线时才算命中
        # 解决重复计数：数字停留时基线追上 → spike 归零
        damage_spike = max(0, current_damage_sum - self.damage_baseline)
        # 指数移动平均更新基线 (0.9 = 缓慢追上，约10帧适应)
        self.damage_baseline = 0.9 * self.damage_baseline + 0.1 * current_damage_sum

        damage_signal = damage_spike / DAMAGE_REWARD_SCALE

        if damage_signal > 0.05:  # 噪声阈值
            reward += min(damage_signal * 0.5, 2.0)
            self.steps_since_last_hit = 0
        else:
            self.steps_since_last_hit += 1
            
        # --- 状态识别 (三维状态) ---
        # 1. 斩位 (Sharpness)
        sharpness_color, _, _ = self.processor.analyze_color_state(self.current_frame, self.sharpness_roi, self.processor.SHARPNESS_COLORS)
        # 2. 气刃槽 (Spirit Gauge - 内槽) - 同时获取颜色(代表等级)和气量(Ratio)
        # 获取颜色 (红刃判定)
        gauge_color, _, _ = self.processor.analyze_color_state(self.current_frame, self.spirit_gauge_roi, self.processor.SPIRIT_COLORS)
        # 获取气量 (通过白线位置)
        gauge_ratio, _ = self.processor.extract_gauge_level(self.current_frame, self.spirit_gauge_roi, gauge_color)

        # [策略引导] 简化逻辑
        # 判断是否红刃 (只看内槽颜色)
        is_red_spirit = (gauge_color in ['red', 'red_2'])
        
        if is_red_spirit:
            # === 红刃状态 (Red Spirit) ===
            # 1. 维持红刃给予持续奖励
            reward += 0.05
            
            # 2. 红刃输出逻辑
            if gauge_ratio > 0.3:
                # A. 气槽充足 (>30%)
                # 策略：最高效输出是 赤刃斩1(Y) 接 气刃斩1(R2) 循环
                # Action 1: Step Slash (Y), Action 6: Spirit I (R2)
                if button_id in [1, 6]:
                    reward += 0.3
                
                # 此时气还很多，不要急着放登龙浪费气
                if button_id in [13, 16]:
                    reward -= 0.1
            else:
                # B. 气槽快用尽 (<=30%)
                # 策略：采用登龙 (Action 13) 或 无双斩 (Action 16) 最大化最后输出
                if button_id in [13, 16]:
                    reward += 0.8 # 给予极高奖励，鼓励收尾
        else:
            # === 非红刃状态 (升刃阶段) ===
            # 策略优先级：居合 > 见切 > 攒气

            # 1. 最高优先级：居合 (Action 12, 17)
            # 用户策略：居合不在乎气槽，且收益高 (升刃/无敌帧)
            if button_id in [12, 17]:
                reward += 0.4
            
            # 2. 次高优先级：见切 (Action 11, 18)
            # 用户策略：只要有一点点气槽就可以见切
            elif button_id == 18:
                # 安全见切 (Action 18) 自带回气，总是值得鼓励
                reward += 0.3
            elif button_id == 11:
                # 普通见切 (Action 11) 需要少量气槽 (>5%) 才有判定
                if gauge_ratio > 0.05:
                    reward += 0.3
                else:
                    reward -= 0.1
            
            # 3. 气刃连段 vs 蓄力攒气
            # 练气槽的多少决定释放几次气刃斩
            elif button_id in [6, 10]: # 气刃斩 (R2)
                # 如果气槽足够支持大回旋前的连段 (假设需要 > 40% 才能打完一套比较舒服)
                if gauge_ratio > 0.4:
                    reward += 0.3
                else:
                    # 气不够最后气刃回旋的时候，只能蓄力气刃斩或普攻攒气
                    # 此时强行 R2 效率低
                    reward -= 0.1
            
            # 4. 蓄力/攒气 (Action 1, 2, 9, 18)
            elif button_id in [1, 2, 9, 18]:
                # 当气槽不足以支撑气刃连段时，鼓励攒气
                if gauge_ratio <= 0.4:
                    reward += 0.3
                # 气满了就别一直蹭刀了，快去开刃
                elif gauge_ratio > 0.9:
                    reward -= 0.05
            
            # 惩罚在非红刃状态下使用终结技
            if button_id in [13, 16]:
                reward -= 0.2

        # [辅助] 气槽管理奖励 (鼓励积攒)
        if gauge_ratio > 0.8:
            reward += 0.01

        # [新增] 斩位维护逻辑
        # 如果斩位掉到 绿/黄/红 (对于上位任务，掉到绿斩通常就需要磨刀了)
        if sharpness_color in ['green', 'yellow', 'red', 'red_2']:
            reward -= 0.02 # 持续焦虑
            # 如果使用了磨刀 (Action 15)
            if button_id == 15:
                reward += 0.3

        # 1. 空挥惩罚
        if is_attacking_step and damage_signal <= 0.05:
            reward -= 0.03 

        # 2. 接近引导: 如果许久 (3秒=30步) 没打中怪，鼓励向前移动 (Move ID 1)
        # 假设 L1 对焦机制有效，向前移动通常意味着接近怪物
        if self.steps_since_last_hit > 30:
            if move_id == 1: # 对应 MOVE_ACTIONS[1] = (0.0, 1.0) 前进
                reward += 0.02
            
        # 等待动作生效 (控制决策频率)
        # 0.1s 意味着大约 10 FPS 的决策速度，对动作游戏来说偏慢，但适合初期训练
        time.sleep(0.1) 
        
        # --- 2. 获取新状态 ---
        raw_frame = self.cap.get_screenshot()
        self.current_frame = raw_frame # 用于 render 显示
        
        # 预处理图像 (灰度、缩放)
        processed_obs = self.processor.preprocess_frame(raw_frame)
        
        # --- 3. 计算奖励 (Reward Shaping) ---
        terminated = False
        truncated = False
        info = {}
        
        # 尝试提取血量 (ROI 需要根据你的分辨率调整!)
        # 假设: 1080p 分辨率下，左上角血条的大致位置
        # 你可能需要运行 utils/vision.py 来校准这个 roi 参数
        # 填入你刚才校准得到的数值 (x, y, w, h)
        health_roi = (149, 74, 705, 28) 
        current_health, _ = self.processor.extract_health_bar(raw_frame, health_roi)
        
        # 基础生存奖励 (鼓励活得更久)
        reward += 0.01 
        
        # --- 血量变化奖励逻辑优化 ---
        health_change = current_health - self.last_health
        
        # 设置阈值 (0.5%) 滤除图像识别的微小噪声
        if health_change < -0.005:
            # 掉血惩罚 (health_change 是负数，所以 reward 减少)
            reward += health_change * 10.0
        elif health_change > 0.005:
            # 回血奖励 (给予高权重，明确告诉 AI 喝药是好事)
            reward += health_change * 20.0
            
        # [新增] 低血量焦虑：如果血量低于 30%，给予持续微小惩罚，迫使它寻找回血手段
        if current_health < 0.3:
            reward -= 0.05

        self.last_health = current_health
        
        # 死亡判定 (血量过低)
        if current_health < 0.05: # 5% 血量以下视为死亡
            terminated = True
            reward -= 10.0 # 死亡大惩罚
            
        return processed_obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置逻辑
        # 在真实训练中，这里应该包含“猫车后等待”或“重新加载任务”的脚本
        # 现在我们简单地重置手柄和内部变量
        self.controller.reset_state()
        
        # 获取初始帧
        raw_frame = self.cap.get_screenshot()
        self.current_frame = raw_frame
        obs = self.processor.preprocess_frame(raw_frame)
        
        # [修正] 重置时读取实际血量，避免开局非满血导致的错误惩罚
        # 这里的 ROI 必须与 step 中保持一致
        health_roi = (149, 74, 705, 28) 
        self.last_health, _ = self.processor.extract_health_bar(raw_frame, health_roi)
        self.steps_since_last_hit = 0
        self.damage_baseline = 0.0   # [尖峰检测] 重置基线

        return obs, {}
        
    def render(self):
        if self.render_mode == 'human' and self.current_frame is not None:
            cv2.imshow("MHWilds RL Agent View", self.current_frame)
            cv2.waitKey(1)
            
    def close(self):
        cv2.destroyAllWindows()