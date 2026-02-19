import vgamepad as vg

# 动作映射表
# 将离散的 Action ID 映射为具体的按键
# 这里的键位基于 Xbox 手柄布局
# L1=LB, R1=RB, L2=LT(Trigger), R2=RT(Trigger)
# post_delay: 动作执行完后的等待时间 (秒)，用于等待动画播放

ACTION_MAP = {
    0: {"name": "No-Op", "duration": 0.05, "post_delay": 0}, # 待机
    
    # ==========================
    #      基础单发招式
    # ==========================
    1: {"name": "Step Slash (Y)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_Y], "duration": 0.1, "post_delay": 0.8},
    2: {"name": "Thrust (B)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "duration": 0.1, "post_delay": 0.6},
    
    # --- 袈裟斩 (位移 + 攒气) ---
    # 后撤袈裟
    3: {"name": "Fade Slash (Back)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_Y, vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "duration": 0.1, "post_delay": 1.0},
    # 左移动袈裟 (左摇杆左 + Y + B)
    4: {"name": "Fade Slash (Left)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_Y, vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "stick_left": (-1.0, 0.0), "duration": 0.1, "post_delay": 1.0},
    # 右移动袈裟 (左摇杆右 + Y + B)
    5: {"name": "Fade Slash (Right)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_Y, vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "stick_left": (1.0, 0.0), "duration": 0.1, "post_delay": 1.0},
    
    # --- 气刃斩 I (R2) ---
    6: {"name": "Spirit I (R2)", "triggers": {"right": 1.0}, "duration": 0.1, "post_delay": 0.8},
    
    # --- 辅助动作 ---
    7: {"name": "Evade (A)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_A], "duration": 0.1, "post_delay": 0.6},
    8: {"name": "Sheathe/Run (RB)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER], "duration": 0.1, "post_delay": 0.5},
    
    # ==========================
    #      太刀核心连招 (Macros)
    # ==========================
    
    # 9: 基础攒气连段 (纵斩 -> 突刺 -> 上挑)
    # 适合蹭刀和积攒练气槽
    9: [
        {"name": "Step Slash", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_Y], "duration": 0.1, "post_delay": 0.8},
        {"name": "Thrust", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "duration": 0.1, "post_delay": 0.6},
        {"name": "Rising Slash", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_Y], "duration": 0.1, "post_delay": 0.8},
    ],

    # 10: 气刃斩 (单次) - R2
    # 拆解原有的固定连招，改为单次执行。允许 AI 在连招间隙穿插垫刀(Y)或见切/居合
    # post_delay 设置为 1.0s 以适应气刃 II/III 的平均节奏
    10: {"name": "Spirit Slash (Next)", "triggers": {"right": 1.0}, "duration": 0.1, "post_delay": 1.0},

    # 11: 见切斩 (突刺 -> R2+B)
    # 为了保证能触发见切（必须接在动作后），我们先垫一个快速突刺
    11: [
        {"name": "Thrust (Setup)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "duration": 0.1, "post_delay": 0.3},
        {"name": "Foresight Slash", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "triggers": {"right": 1.0}, "duration": 0.2, "post_delay": 1.5},
    ],
    
    # 12: 居合气刃斩 (大居合)
    # 连招：纵斩(垫刀) -> 特殊纳刀(R2+A) -> 居合气刃斩(R2)
    12: [
        {"name": "Step Slash (Setup)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_Y], "duration": 0.1, "post_delay": 0.5},
        {"name": "Special Sheathe", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_A], "triggers": {"right": 1.0}, "duration": 0.2, "post_delay": 1.0},
        {"name": "Iai Spirit Slash", "triggers": {"right": 1.0}, "duration": 0.1, "post_delay": 1.5},
    ],
    
    # 13: 气刃兜割 (登龙斩) - RB + Y
    # 消耗一级气刃等级，打出多段伤害
    13: {"name": "Helm Breaker", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER, vg.XUSB_BUTTON.XUSB_GAMEPAD_Y], "duration": 0.2, "post_delay": 2.0},

    # 14: 集中弱点攻击 (Focus Weakness Attack) - LT + RB
    # 荒野新机制：按住 LT 瞄准伤口，按 RB 攻击
    14: {"name": "Focus Weakness Attack", "triggers": {"left": 1.0}, "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER], "duration": 0.5, "post_delay": 1.5},

    # 15: 喝药宏 (收刀 -> 喝药)
    15: [
        {"name": "Sheathe", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER], "duration": 0.1, "post_delay": 0.5},
        {"name": "Use Item", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_X], "duration": 0.1, "post_delay": 2.0},
    ],

    # 16: 练气解放无双斩 (Spirit Release Flurry) - RT + Y
    # 红刃状态下的终极爆发，消耗气刃等级
    16: {"name": "Spirit Release Flurry", "triggers": {"right": 1.0}, "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_Y], "duration": 0.2, "post_delay": 3.5},

    # 17: 连续居合 (Continuous Iai)
    # 突刺 -> 特殊纳刀 -> 大居合 -> 再次特殊纳刀 (取消硬直保持反击态)
    17: [
        {"name": "Thrust (Setup)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "duration": 0.1, "post_delay": 0.3},
        {"name": "Special Sheathe", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_A], "triggers": {"right": 1.0}, "duration": 0.2, "post_delay": 0.8},
        
        # --- 优化版：利用输入缓冲 (Input Buffering) ---
        # 1. 释放大居合
        {"name": "Iai Spirit Slash", "triggers": {"right": 1.0}, "duration": 0.1, "post_delay": 0.0}, 
        # 2. 缓冲阶段：在大居合动作期间，按住 RT 不放 (release=False)
        # 这里的 duration 模拟了硬直时间，期间 RT 一直被按住，游戏会缓冲这个输入
        {"name": "Buffer RT", "triggers": {"right": 1.0}, "duration": 0.5, "release": False, "post_delay": 0.0},
        # 3. 触发纳刀：在硬直结束瞬间，补按 A (此时 RT 依然是按下的)
        {"name": "Re-Sheathe (Trigger)", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_A], "triggers": {"right": 1.0}, "duration": 0.1, "release": True, "post_delay": 1.0},
    ],

    # 18: 安全见切 (Safe Foresight) - 空槽对策
    # 逻辑：按住 RT (气刃 I) 瞬间回气 -> 立即接 见切 (RT+B)
    # 适用于气槽为空但急需无敌帧的情况
    18: [
        # 按住 RT 0.3秒 (蹭气/蓄力)，不松开
        {"name": "Spirit Regen (Hold)", "triggers": {"right": 1.0}, "duration": 0.3, "release": False, "post_delay": 0.0},
        # 紧接着按下 B (此时 RT 仍按下 -> 触发 RT+B 见切)
        {"name": "Foresight Slash", "buttons": [vg.XUSB_BUTTON.XUSB_GAMEPAD_B], "triggers": {"right": 1.0}, "duration": 0.2, "release": True, "post_delay": 1.5},
    ]
}

# 摇杆动作 (简化版)
# 我们可以将移动也作为离散动作的一部分，或者使用 MultiDiscrete 空间
# 这里为了简单，先定义几个固定的移动方向
MOVE_ACTIONS = {
    0: (0.0, 0.0),   # 停止
    1: (0.0, 1.0),   # 前
    2: (0.0, -1.0),  # 后
    3: (-1.0, 0.0),  # 左
    4: (1.0, 0.0),   # 右
}

# 组合动作示例 (如果需要)
# 例如：拔刀斩 (RB + Y)
# COMBO_ACTIONS = {
#     ...
# }