"""
综合实时测试工具：同时测试伤害输出检测和受伤检测。

在屏幕上实时显示:
- 伤害数字检测结果 (detect_hit_signals)
- 血条检测结果 (extract_health_bar)
- 各项指标的统计数据

用法:
    python tools/test_detection_live.py
    python tools/test_detection_live.py --window "Monster Hunter Wilds"

控制键:
    G     - 暂停/继续画面
    T     - 切换触发模式 (仅攻击后检测 vs 持续检测)
    D     - 切换详细信息面板
    R     - 重置统计数据
    ;     - 退出
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ctypes

from utils.capture import WindowCapture
from utils.vision import ImageProcessor


# ──────────────────────────────────────────────
#  XInput 手柄读取 (Windows 原生)
# ──────────────────────────────────────────────
class _XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]

class _XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", _XINPUT_GAMEPAD),
    ]

XINPUT_BUTTON_Y = 0x8000
XINPUT_BUTTON_B = 0x2000
XINPUT_BUTTON_A = 0x1000
RT_THRESHOLD = 100

def _load_xinput():
    for lib in ["xinput1_4", "xinput1_3", "xinput9_1_0"]:
        try:
            return ctypes.windll.LoadLibrary(lib)
        except OSError:
            continue
    return None

_xinput_dll = _load_xinput()

def get_gamepad_state(controller_id=None):
    """
    读取手柄状态，返回 (buttons_mask, right_trigger_value, controller_id) 或 None。
    如果 controller_id=None，自动扫描 0-3 找第一个连接的手柄。
    """
    if _xinput_dll is None:
        return None

    ids_to_check = [controller_id] if controller_id is not None else range(4)
    state = _XINPUT_STATE()
    for cid in ids_to_check:
        res = _xinput_dll.XInputGetState(cid, ctypes.byref(state))
        if res == 0:
            return state.Gamepad.wButtons, state.Gamepad.bRightTrigger, cid
    return None


def detect_all_controllers():
    """扫描所有 4 个 XInput 槽位，返回已连接的 controller_id 列表"""
    if _xinput_dll is None:
        return []
    connected = []
    state = _XINPUT_STATE()
    for cid in range(4):
        res = _xinput_dll.XInputGetState(cid, ctypes.byref(state))
        if res == 0:
            connected.append(cid)
    return connected


# ──────────────────────────────────────────────
#  默认 ROI (与 game_env.py 保持一致)
# ──────────────────────────────────────────────
DAMAGE_ROI = (924, 25, 1534, 1398)
HEALTH_ROI = (149, 74, 705, 28)


class DetectionTester:
    """综合检测测试器"""

    def __init__(self, window_name, damage_roi, health_roi, controller_id=1):
        self.cap = WindowCapture(window_name)
        self.processor = ImageProcessor(
            crnn_model_path="models/crnn/damage_crnn_best.pth"
        )
        self.damage_roi = damage_roi
        self.health_roi = health_roi
        self.forced_pad_id = controller_id

        # 状态
        self.paused = False
        self.trigger_mode = False
        self.show_detail = True
        self.current_frame = None

        # 统计 - 伤害输出
        self.dmg_history = deque(maxlen=50)       # 最近50次检测结果
        self.dmg_event_log = deque(maxlen=12)     # 显示在画面上的事件日志
        self.dmg_total_frames = 0                 # 总检测帧数
        self.dmg_total_hits = 0                   # 总检测到命中的帧数
        self.dmg_total_damage = 0                 # 累计伤害值
        self.dmg_fps_times = deque(maxlen=30)     # 检测耗时 (ms)
        self.dmg_false_positive_suspect = 0       # 疑似误报 (无攻击时检测到)

        # 统计 - 血条
        self.hp_history = deque(maxlen=300)       # 最近300帧血量 (约30秒)
        self.hp_last = None                       # 上一帧血量
        self.hp_drops = []                        # 掉血事件 [(时间, 幅度)]
        self.hp_heals = []                        # 回血事件 [(时间, 幅度)]
        self.hp_stable_count = 0                  # 连续稳定帧数
        self.hp_noise_events = 0                  # 噪声事件 (微小波动)
        self.hp_fps_times = deque(maxlen=30)      # 检测耗时 (ms)

        # 触发模式相关
        self.last_activity_time = time.time()
        self.trigger_window = 3.0  # 秒

        # 手柄状态 (边缘检测用)
        self._prev_y = False
        self._prev_b = False
        self._prev_rt = False
        self.pad_connected = False
        self.pad_id = None          # 锁定的手柄槽位 (0-3)
        self.pad_last_action = ""   # 最近触发的按键名

    def reset_stats(self):
        """重置所有统计数据"""
        self.dmg_history.clear()
        self.dmg_event_log.clear()
        self.dmg_total_frames = 0
        self.dmg_total_hits = 0
        self.dmg_total_damage = 0
        self.dmg_fps_times.clear()
        self.dmg_false_positive_suspect = 0

        self.hp_history.clear()
        self.hp_last = None
        self.hp_drops.clear()
        self.hp_heals.clear()
        self.hp_stable_count = 0
        self.hp_noise_events = 0
        self.hp_fps_times.clear()
        print("[统计已重置]")

    def detect_damage(self, frame):
        """运行伤害输出检测并更新统计"""
        t0 = time.perf_counter()
        result = self.processor.detect_hit_signals(
            frame, self.damage_roi, max_ocr=10, debug=True
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.dmg_fps_times.append(elapsed_ms)

        total_damage, hit_count, mask, filtered_mask, ocr_details, rejected = result
        self.dmg_total_frames += 1

        if hit_count > 0:
            self.dmg_total_hits += 1
            self.dmg_total_damage += total_damage

            values = [d['value'] for d in ocr_details if d['value'] is not None]
            timestamp = time.strftime("%H:%M:%S")
            self.dmg_event_log.append(
                f"[{timestamp}] HIT x{hit_count} = {values} (sum={total_damage})"
            )

        self.dmg_history.append({
            'hit_count': hit_count,
            'total_damage': total_damage,
            'details': ocr_details,
            'elapsed_ms': elapsed_ms,
            'rejected': rejected,
        })

        return total_damage, hit_count, mask, filtered_mask, ocr_details, rejected

    def detect_health(self, frame):
        """运行血条检测并更新统计"""
        t0 = time.perf_counter()
        ratio, mask = self.processor.extract_health_bar(frame, self.health_roi)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.hp_fps_times.append(elapsed_ms)

        self.hp_history.append(ratio)

        if self.hp_last is not None:
            change = ratio - self.hp_last
            if change < -0.005:
                # 掉血
                self.hp_drops.append((time.time(), change))
                timestamp = time.strftime("%H:%M:%S")
                self.dmg_event_log.append(
                    f"[{timestamp}] DAMAGE TAKEN: HP {self.hp_last:.1%} -> {ratio:.1%} ({change:+.1%})"
                )
            elif change > 0.005:
                # 回血
                self.hp_heals.append((time.time(), change))
                timestamp = time.strftime("%H:%M:%S")
                self.dmg_event_log.append(
                    f"[{timestamp}] HEALED: HP {self.hp_last:.1%} -> {ratio:.1%} ({change:+.1%})"
                )
            elif abs(change) > 0.001:
                # 噪声
                self.hp_noise_events += 1
            else:
                self.hp_stable_count += 1

        self.hp_last = ratio
        return ratio, mask

    def draw_overlay(self, frame, dmg_result, hp_result):
        """在画面上绘制所有检测结果的叠加层"""
        display = frame.copy()
        h_frame, w_frame = display.shape[:2]

        total_damage, hit_count, dmg_mask, filtered_mask, ocr_details, rejected = dmg_result
        hp_ratio, hp_mask = hp_result

        # ── 绘制 ROI 边框 ──
        dx, dy, dw, dh = self.damage_roi
        cv2.rectangle(display, (dx, dy), (dx + dw, dy + dh), (0, 255, 255), 3)
        cv2.putText(display, "Damage ROI", (dx, dy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        hx, hy, hw, hh = self.health_roi
        cv2.rectangle(display, (hx, hy), (hx + hw, hy + hh), (0, 255, 0), 3)
        cv2.putText(display, "Health ROI", (hx + hw + 10, hy + hh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # ── 绘制伤害检测框 ──
        for det in ocr_details:
            bx, by, bw, bh = det['bbox']
            abs_x, abs_y = dx + bx, dy + by
            if det['value'] is not None:
                cv2.rectangle(display, (abs_x, abs_y),
                              (abs_x + bw, abs_y + bh), (0, 255, 0), 3)
                cv2.putText(display, str(det['value']),
                            (abs_x, abs_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            else:
                cv2.rectangle(display, (abs_x, abs_y),
                              (abs_x + bw, abs_y + bh), (0, 0, 255), 1)

        # ── 信息面板 (左上角) ──
        panel_x, panel_y = 15, 30
        line_h = 36

        def put(text, color=(255, 255, 255), bold=False):
            nonlocal panel_y
            thickness = 3 if bold else 2
            cv2.putText(display, text, (panel_x, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, thickness)
            panel_y += line_h

        # 标题
        put("=== DETECTION TEST ===", (0, 255, 255), bold=True)

        # 模式状态
        mode_text = "TRIGGER" if self.trigger_mode else "CONTINUOUS"
        mode_color = (0, 200, 255) if self.trigger_mode else (0, 255, 0)
        put(f"Mode: {mode_text}  |  {'PAUSED' if self.paused else 'LIVE'}",
            mode_color)

        # 手柄状态
        if self.pad_connected:
            pad = get_gamepad_state(self.pad_id)
            if pad:
                buttons, rt_val, _ = pad
                y_st = "ON" if buttons & XINPUT_BUTTON_Y else "--"
                b_st = "ON" if buttons & XINPUT_BUTTON_B else "--"
                rt_st = f"{rt_val:3d}"
                put(f"Pad: Y={y_st} B={b_st} RT={rt_st}  Last={self.pad_last_action}",
                    (0, 255, 0))
            else:
                put("Pad: Disconnected", (0, 0, 255))
        else:
            put("Pad: Not found", (100, 100, 100))

        panel_y += 5

        # ── 血条状态 ──
        put("--- HP Detection ---", (0, 255, 0), bold=True)
        hp_color = (0, 255, 0) if hp_ratio > 0.5 else (0, 255, 255) if hp_ratio > 0.2 else (0, 0, 255)
        put(f"HP: {hp_ratio:.1%}", hp_color, bold=True)

        # 血条可视化条
        bar_x = panel_x
        bar_y = panel_y
        bar_w = 320
        bar_h = 20
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        fill_w = int(bar_w * min(1.0, max(0.0, hp_ratio)))
        if fill_w > 0:
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          hp_color, -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (255, 255, 255), 1)
        panel_y += line_h

        if self.show_detail:
            hp_ms = np.mean(list(self.hp_fps_times)) if self.hp_fps_times else 0
            put(f"  Latency: {hp_ms:.1f}ms", (180, 180, 180))
            put(f"  Drops: {len(self.hp_drops)}  Heals: {len(self.hp_heals)}",
                (180, 180, 180))
            put(f"  Noise: {self.hp_noise_events}  Stable: {self.hp_stable_count}",
                (180, 180, 180))

            # 血量稳定性评估
            if len(self.hp_history) > 30:
                recent = list(self.hp_history)[-30:]
                std = np.std(recent)
                stability = "STABLE" if std < 0.005 else "NOISY" if std < 0.02 else "UNSTABLE"
                stab_color = ((0, 255, 0) if stability == "STABLE"
                              else (0, 255, 255) if stability == "NOISY"
                              else (0, 0, 255))
                put(f"  Stability: {stability} (std={std:.4f})", stab_color)

        panel_y += 5

        # ── 伤害检测状态 ──
        put("--- DMG Detection ---", (0, 255, 255), bold=True)
        if hit_count > 0:
            put(f"HIT! x{hit_count} = {total_damage}", (0, 255, 0), bold=True)
        else:
            put("No hits", (128, 128, 128))

        if self.show_detail:
            dmg_ms = np.mean(list(self.dmg_fps_times)) if self.dmg_fps_times else 0
            put(f"  Latency: {dmg_ms:.1f}ms", (180, 180, 180))
            put(f"  Total frames: {self.dmg_total_frames}", (180, 180, 180))
            hit_rate = (self.dmg_total_hits / self.dmg_total_frames * 100
                        if self.dmg_total_frames > 0 else 0)
            put(f"  Hit frames: {self.dmg_total_hits} ({hit_rate:.1f}%)",
                (180, 180, 180))
            put(f"  Cumulative DMG: {self.dmg_total_damage}", (180, 180, 180))

            if rejected:
                put(f"  Rejected: area_s={rejected['area_small']} "
                    f"area_b={rejected['area_big']} "
                    f"asp={rejected['aspect']}", (120, 120, 120))

        panel_y += 5

        # ── 事件日志 ──
        put("--- Event Log ---", (200, 200, 200), bold=True)
        for entry in list(self.dmg_event_log)[-8:]:
            color = (180, 180, 180)
            if "HIT" in entry:
                color = (0, 255, 0)
            elif "DAMAGE TAKEN" in entry:
                color = (0, 100, 255)
            elif "HEALED" in entry:
                color = (0, 255, 200)
            put(f"  {entry}", color)

        # ── 血量历史曲线 (右下角) ──
        if len(self.hp_history) > 2:
            self._draw_hp_graph(display, w_frame, h_frame)

        # ── 操作提示 (底部) ──
        help_y = h_frame - 15
        cv2.putText(display,
                    "G=Pause  T=TriggerMode  D=Detail  R=Reset  ;=Quit",
                    (15, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (150, 150, 150), 2)

        return display

    def _draw_hp_graph(self, display, w_frame, h_frame):
        """绘制血量历史折线图"""
        graph_w = 400
        graph_h = 120
        margin = 15
        gx = w_frame - graph_w - margin
        gy = h_frame - graph_h - margin - 25

        # 半透明背景
        overlay = display.copy()
        cv2.rectangle(overlay, (gx - 5, gy - 20),
                      (gx + graph_w + 5, gy + graph_h + 5),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)

        cv2.putText(display, "HP History (30s)", (gx, gy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # 画网格线
        for pct in [0.25, 0.5, 0.75]:
            y_line = gy + int((1.0 - pct) * graph_h)
            cv2.line(display, (gx, y_line), (gx + graph_w, y_line),
                     (50, 50, 50), 1)

        # 画血量曲线
        data = list(self.hp_history)
        n = len(data)
        if n < 2:
            return

        points = []
        for i, val in enumerate(data):
            px = gx + int(i / max(n - 1, 1) * graph_w)
            py = gy + int((1.0 - min(1.0, max(0.0, val))) * graph_h)
            points.append((px, py))

        pts = np.array(points, dtype=np.int32)
        # 根据最新血量选颜色
        latest = data[-1]
        color = ((0, 255, 0) if latest > 0.5
                 else (0, 255, 255) if latest > 0.2
                 else (0, 0, 255))
        cv2.polylines(display, [pts], False, color, 2)

        # 标注当前值
        cv2.putText(display, f"{latest:.1%}",
                    (gx + graph_w - 60, gy + graph_h + 20 if latest > 0.1 else gy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def run(self):
        """主循环"""
        print("\n" + "=" * 55)
        print("  综合检测实时测试")
        print("=" * 55)
        print(f"  伤害检测 ROI: {self.damage_roi}")
        print(f"  血条检测 ROI: {self.health_roi}")
        print()
        print("  操作说明:")
        print("    G - 暂停/继续画面")
        print("    T - 切换触发模式 (持续检测 / 仅攻击后检测)")
        print("    D - 显示/隐藏详细信息")
        print("    R - 重置统计数据")
        print("    ; - 退出")
        print("=" * 55 + "\n")

        last_key_time = 0
        KEY_COOLDOWN = 0.3

        # 检测手柄
        connected = detect_all_controllers()
        print(f"  已连接的手柄槽位: {connected if connected else '无'}")
        self.pad_id = self.forced_pad_id
        if self.pad_id in connected:
            self.pad_connected = True
            print(f"  [OK] 使用 controller #{self.pad_id}，按 Y/B/RT 触发伤害检测窗口")
        else:
            print(f"  [!] controller #{self.pad_id} 未连接! 尝试 --controller 参数指定其他槽位")

        cv2.namedWindow("Detection Test (;=Quit)", cv2.WINDOW_NORMAL)

        while True:
            now = time.time()

            # ── 轮询手柄 ──
            attack_triggered = False
            pad = get_gamepad_state(self.pad_id)
            if pad is not None:
                self.pad_connected = True
                buttons, rt_val, _ = pad

                y_now = bool(buttons & XINPUT_BUTTON_Y)
                b_now = bool(buttons & XINPUT_BUTTON_B)
                rt_now = rt_val > RT_THRESHOLD

                # 边缘检测: 从未按到按下的瞬间才触发
                if y_now and not self._prev_y:
                    attack_triggered = True
                    self.pad_last_action = "Y"
                if b_now and not self._prev_b:
                    attack_triggered = True
                    self.pad_last_action = "B"
                if rt_now and not self._prev_rt:
                    attack_triggered = True
                    self.pad_last_action = "RT"

                self._prev_y = y_now
                self._prev_b = b_now
                self._prev_rt = rt_now
            else:
                self.pad_connected = False

            # 手柄攻击 → 刷新活动时间
            if attack_triggered:
                self.last_activity_time = now
                timestamp = time.strftime("%H:%M:%S")
                self.dmg_event_log.append(
                    f"[{timestamp}] ATTACK: {self.pad_last_action} pressed"
                )

            # 抓帧
            if not self.paused:
                self.current_frame = self.cap.get_screenshot()

            frame = self.current_frame
            if frame is None:
                time.sleep(0.01)
                continue

            # 判断是否处于活动窗口
            is_active = True
            if self.trigger_mode:
                elapsed = now - self.last_activity_time
                if elapsed > self.trigger_window:
                    is_active = False

            # ── 运行检测 ──
            if is_active and not self.paused:
                dmg_result = self.detect_damage(frame)
                hp_result = self.detect_health(frame)
            else:
                # 空闲时仍检测血量 (开销很低)
                hp_result = self.detect_health(frame)
                dmg_result = (0, 0, None, None, [], None)

            # ── 绘制叠加层 ──
            display = self.draw_overlay(frame, dmg_result, hp_result)

            cv2.imshow("Detection Test (;=Quit)", display)

            # 显示血条 mask (小窗口)
            if hp_result[1] is not None:
                hp_mask = hp_result[1]
                # 放大以便观察
                if hp_mask.shape[1] < 400:
                    scale = 400 / hp_mask.shape[1]
                    hp_mask = cv2.resize(hp_mask, None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_NEAREST)
                cv2.imshow("HP Mask (white=health)", hp_mask)

            # ── 按键处理 ──
            key = cv2.waitKey(1) & 0xFF
            if key == ord(';'):
                break
            elif key == ord('g') and (now - last_key_time > KEY_COOLDOWN):
                self.paused = not self.paused
                print(f"  画面已{'暂停' if self.paused else '继续'}")
                last_key_time = now
            elif key == ord('t') and (now - last_key_time > KEY_COOLDOWN):
                self.trigger_mode = not self.trigger_mode
                print(f"  触发模式: {'开启' if self.trigger_mode else '关闭'}")
                last_key_time = now
            elif key == ord('d') and (now - last_key_time > KEY_COOLDOWN):
                self.show_detail = not self.show_detail
                last_key_time = now
            elif key == ord('r') and (now - last_key_time > KEY_COOLDOWN):
                self.reset_stats()
                last_key_time = now

        # ── 退出时打印统计 ──
        self._print_summary()
        cv2.destroyAllWindows()

    def _print_summary(self):
        """退出时打印统计摘要"""
        print("\n" + "=" * 55)
        print("  检测测试统计摘要")
        print("=" * 55)

        print("\n  --- 伤害输出检测 (Damage Done) ---")
        print(f"  总检测帧数: {self.dmg_total_frames}")
        if self.dmg_total_frames > 0:
            hit_rate = self.dmg_total_hits / self.dmg_total_frames * 100
            print(f"  命中帧数:   {self.dmg_total_hits} ({hit_rate:.1f}%)")
        print(f"  累计伤害值: {self.dmg_total_damage}")
        if self.dmg_fps_times:
            print(f"  检测延迟:   {np.mean(list(self.dmg_fps_times)):.1f}ms "
                  f"(P95={np.percentile(list(self.dmg_fps_times), 95):.1f}ms)")

        print("\n  --- 受伤检测 (Damage Taken / HP) ---")
        if self.hp_history:
            print(f"  总检测帧数: {len(self.hp_history)}")
            print(f"  最终血量:   {list(self.hp_history)[-1]:.1%}")
            print(f"  掉血事件:   {len(self.hp_drops)} 次")
            for t, delta in self.hp_drops[-5:]:
                print(f"    {time.strftime('%H:%M:%S', time.localtime(t))}: {delta:+.1%}")
            print(f"  回血事件:   {len(self.hp_heals)} 次")
            print(f"  噪声波动:   {self.hp_noise_events} 次")

            if len(self.hp_history) > 30:
                std = np.std(list(self.hp_history)[-30:])
                print(f"  最近稳定性: std={std:.4f} "
                      f"({'良好' if std < 0.005 else '有噪声' if std < 0.02 else '不稳定'})")

        if self.hp_fps_times:
            print(f"  检测延迟:   {np.mean(list(self.hp_fps_times)):.1f}ms")

        print("\n" + "=" * 55)

        # 保存到 JSON
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        summary = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'damage_detection': {
                'total_frames': self.dmg_total_frames,
                'hit_frames': self.dmg_total_hits,
                'hit_rate': (self.dmg_total_hits / self.dmg_total_frames
                             if self.dmg_total_frames > 0 else 0),
                'cumulative_damage': self.dmg_total_damage,
                'avg_latency_ms': (float(np.mean(list(self.dmg_fps_times)))
                                   if self.dmg_fps_times else 0),
            },
            'health_detection': {
                'total_frames': len(self.hp_history),
                'final_hp': float(list(self.hp_history)[-1]) if self.hp_history else 0,
                'drop_events': len(self.hp_drops),
                'heal_events': len(self.hp_heals),
                'noise_events': self.hp_noise_events,
                'avg_latency_ms': (float(np.mean(list(self.hp_fps_times)))
                                   if self.hp_fps_times else 0),
            },
        }

        path = os.path.join(logs_dir, "detection_test_summary.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  统计已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="综合检测实时测试")
    parser.add_argument("--window", type=str, default="Monster Hunter Wilds",
                        help="游戏窗口名称")
    parser.add_argument("--damage-roi", type=str, default=None,
                        help="伤害检测ROI (x,y,w,h)")
    parser.add_argument("--health-roi", type=str, default=None,
                        help="血条检测ROI (x,y,w,h)")
    parser.add_argument("--controller", type=int, default=1,
                        help="XInput 手柄槽位 (0-3, 默认=1)")
    args = parser.parse_args()

    damage_roi = DAMAGE_ROI
    health_roi = HEALTH_ROI

    if args.damage_roi:
        damage_roi = tuple(int(v) for v in args.damage_roi.split(','))
    if args.health_roi:
        health_roi = tuple(int(v) for v in args.health_roi.split(','))

    tester = DetectionTester(args.window, damage_roi, health_roi,
                             controller_id=args.controller)
    tester.run()


if __name__ == "__main__":
    main()
