import sys
import os
import cv2
import time
import keyboard
import numpy as np
import ctypes

# 将项目根目录添加到 python path，以便导入 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.capture import WindowCapture
from utils.vision import ImageProcessor

# ============ XInput 真实手柄读取 (Windows 原生, 无需安装) ============
class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", XINPUT_GAMEPAD),
    ]

# Xbox 按键掩码
XINPUT_BUTTON_Y = 0x8000
XINPUT_BUTTON_B = 0x2000
XINPUT_BUTTON_A = 0x1000
XINPUT_BUTTON_X = 0x4000
XINPUT_BUTTON_RB = 0x0200  # Right Shoulder (R1)

# RT 触发阈值 (0-255, 超过此值视为按下)
RT_THRESHOLD = 100

def _load_xinput():
    """加载 XInput DLL"""
    for lib in ["xinput1_4", "xinput1_3", "xinput9_1_0"]:
        try:
            return ctypes.windll.LoadLibrary(lib)
        except OSError:
            continue
    return None

_xinput_dll = _load_xinput()

def get_gamepad_state(controller_id=0):
    """
    读取真实手柄状态
    :return: (buttons_mask, right_trigger_value) 或 None (手柄未连接)
    """
    if _xinput_dll is None:
        return None
    state = XINPUT_STATE()
    res = _xinput_dll.XInputGetState(controller_id, ctypes.byref(state))
    if res == 0:  # ERROR_SUCCESS
        return state.Gamepad.wButtons, state.Gamepad.bRightTrigger
    return None

# 全局变量用于存储当前帧，供鼠标回调使用
mouse_frame = None

def mouse_click(event, x, y, flags, param):
    global mouse_frame
    if event == cv2.EVENT_LBUTTONDOWN and mouse_frame is not None:
        # 转换点击点的颜色为 HSV
        hsv = cv2.cvtColor(mouse_frame, cv2.COLOR_BGR2HSV)
        if y < hsv.shape[0] and x < hsv.shape[1]:
            # 将 numpy uint8 转换为普通 int，防止减法溢出 (例如 0 - 10 = 246)
            val = hsv[y, x].astype(int)
            print(f"\n🔍 [取色器] 点击位置: ({x}, {y}) | HSV: {val}")
            print(f"    >> 建议 Lower: np.array([{max(0, val[0]-10)}, {max(0, val[1]-40)}, {max(0, val[2]-40)}])")
            print(f"    >> 建议 Upper: np.array([{min(180, val[0]+10)}, 255, 255])")

def calibrate():
    print("正在初始化屏幕捕获...")
    try:
        # 尝试捕获游戏窗口，如果失败则捕获全屏
        cap = WindowCapture("Monster Hunter Wilds")
    except:
        cap = WindowCapture("Notepad") # 仅供测试用

    processor = ImageProcessor()

    print("\n" + "="*50)
    print("【模式选择】")
    print("1. 血条 (Health Bar) - 颜色识别")
    print("2. 数字 (Digits) - OCR识别 (如伤害统计/道具数)")
    print("3. 伤害检测 (Hit Detection) - 动态色块识别 [推荐]")
    print("4. 斩位 (Sharpness) - 武器锋利度 (小刀图标)")
    print("5. 练气槽 (Spirit Gauge) - 颜色(红刃判定) & 气量")
    mode = input("请输入序号 (1-5): ").strip()

    print("\n" + "="*50)
    print("【步骤 1: 捕获画面】")
    print("正在显示实时画面...")
    print("请切换回游戏进行操作（如攻击木桩）。")
    print(">>> 当看到伤害数字时，按键盘【G】键冻结画面 <<<")
    print("按【;】键退出程序。")
    print("="*50 + "\n")

    frame = None
    while True:
        frame = cap.get_screenshot()

        # 显示提示
        display_frame = frame.copy()
        cv2.putText(display_frame, "Live View: Press 'G' to Freeze", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Live View", display_frame)

        if cv2.waitKey(1) & 0xFF == ord(';'):
            cv2.destroyAllWindows()
            return

        if keyboard.is_pressed('g'):
            print("检测到 G 键，画面已冻结！")
            break
        if keyboard.is_pressed(';'):
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Live View")

    print("\n" + "="*50)
    print("【步骤 2: 校准 ROI】")
    print("1. 弹窗后，请用鼠标左键【框选】目标区域。")
    print("2. 选好后，按【SPACE】或【ENTER】确认。")
    print("3. 如果想取消，按【c】。")
    print("="*50 + "\n")

    # 调用 OpenCV 的 ROI 选择器
    # 返回格式: (x, y, w, h)
    if mode == '2':
        win_name = "Calibrate Digits"
    elif mode == '3':
        win_name = "Calibrate Hit Area"
    elif mode == '4':
        win_name = "Calibrate Sharpness"
    elif mode == '5':
        win_name = "Calibrate Spirit Gauge"
    else:
        win_name = "Calibrate Health Bar"
    roi = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win_name)

    # 如果用户取消了选择 (返回全是0)
    if roi == (0, 0, 0, 0):
        print("未选择区域，程序退出。")
        return

    print(f"\n✅ 校准完成！")
    print(f"你的 ROI 坐标是: {roi}")
    print(f"格式为: (x, y, w, h)")
    print("-" * 30)
    if mode == '1':
        print(f"请更新 envs/game_env.py 中的 health_roi")
    elif mode == '3':
        print(f"请更新 envs/game_env.py 中的 damage_roi (这是一个大范围区域)")
    elif mode == '4':
        print(f"请更新 envs/game_env.py 中的 sharpness_roi")
    elif mode == '5':
        print(f"请更新 envs/game_env.py 中的 spirit_gauge_roi")
    else:
        print(f"请记录此坐标用于 OCR (如 damage_roi)")
    print("-" * 30)

    # 实时验证环节
    print("\n正在开启实时验证模式 (按 ';' 退出)...")
    print(">>> 按 'G' 键暂停/恢复画面 (方便取色) <<<")
    if mode == '1':
        print("观察 'Health Mask': 白色代表识别到的血量。")
    elif mode == '3':
        print("观察 'Hit Mask': 当出现伤害数字时，应该出现白色块。")
        if _xinput_dll is not None:
            pad_test = get_gamepad_state()
            if pad_test is not None:
                print(">>> 已检测到真实手柄! 按手柄 [Y] 或 [RT] 攻击时自动触发短窗口检测 <<<")
            else:
                print("  [!] XInput 可用但未检测到手柄，请插入手柄")
        else:
            print("  [!] XInput DLL 加载失败，手柄触发不可用")
        DETECT_DELAY = 0.5   # 攻击后等多久再开始检测 (等动画命中，踏步斩需要~0.5s)
        DETECT_WINDOW = 1.5  # 检测窗口持续时间 (秒，覆盖伤害数字完整生命周期)
        # 用于追踪按键的"按下边缘" (防止长按连续触发)
        _prev_y_pressed = False
        _prev_rt_pressed = False
    elif mode == '4':
        print("观察画面: 显示颜色及填充率 (Ratio)")
    elif mode == '5':
        print("观察画面: 显示颜色及填充率 (Ratio)")
    else:
        print("观察 'OCR Debug': 必须黑白分明且文字清晰，否则 OCR 会失败。")

    # 设置鼠标回调函数
    cv2.namedWindow("Verification View")
    cv2.setMouseCallback("Verification View", mouse_click)

    paused = False
    last_pause_time = 0
    current_frame = cap.get_screenshot() # 初始化一帧
    damage_baseline = 0.0  # [尖峰检测] 用于 mode 3

    # 历史记录相关
    damage_history = []
    last_record_time = 0

    # --- 触发模式 (Trigger Mode) ---
    # 用于模拟实际运行时的性能优化策略：仅在操作发生后的一段时间内进行检测
    trigger_mode = False
    TRIGGER_WINDOW = 3.0 # 窗口时间设为 3.0秒，以覆盖气刃兜割/无双解放等慢速动作
    last_act_time = time.time()

    def on_key_event(e):
        nonlocal last_act_time
        if e.name not in ['g', 'h', ';', 't']: # 排除控制键
            last_act_time = time.time()
    keyboard.hook(on_key_event)

    while True:
        global mouse_frame

        # 暂停控制 (防止按一次键触发多次，增加 0.3s 冷却)
        if keyboard.is_pressed('g') and (time.time() - last_pause_time > 0.3):
            paused = not paused
            print(f"验证画面已 {'暂停' if paused else '继续'}")
            last_pause_time = time.time()
        
        # 切换触发模式 't'
        if keyboard.is_pressed('t') and (time.time() - last_pause_time > 0.3):
            trigger_mode = not trigger_mode
            print(f"\n⚡ 触发模式 (Trigger Mode): {'[开启]' if trigger_mode else '[关闭]'}")
            print(f"   >>> 仅在按键活动后 {TRIGGER_WINDOW}秒内检测 (模拟 RT 触发)")
            last_pause_time = time.time()

        if not paused:
            current_frame = cap.get_screenshot()

        # 使用副本进行绘制，避免污染原始帧 (特别是暂停时)
        display_frame = current_frame.copy()

        # 在副本上画框
        x, y, w, h = roi
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # 检查是否处于活动窗口期
        is_active = True
        if trigger_mode:
            # 模拟: 按下或松开后 TRIGGER_WINDOW 内有效
            time_left = TRIGGER_WINDOW - (time.time() - last_act_time)
            if time_left < 0:
                is_active = False
                time_left = 0
            
            status_text = "Active" if is_active else "Idle (Save Perf)"
            col = (0, 255, 0) if is_active else (128, 128, 128)
            cv2.putText(display_frame, f"Trigger: {status_text}", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            
            # 绘制倒计时条
            if is_active:
                bar_width = int((time_left / TRIGGER_WINDOW) * 200)
                cv2.rectangle(display_frame, (20, 80), (20 + bar_width, 85), col, -1)
                cv2.rectangle(display_frame, (20, 80), (20 + 200, 85), (255, 255, 255), 1)

        if mode == '1':
            if is_active:
                ratio, mask = processor.extract_health_bar(current_frame, roi)
                cv2.putText(display_frame, f"Health: {ratio:.1%}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow("Health Mask (White=Health)", mask)
        elif mode == '3':
            DAMAGE_SCALE = 50.0

            # --- 读取真实手柄状态，检测按键边缘 (按下瞬间) ---
            attack_triggered = False
            attack_name = ""
            pad = get_gamepad_state()
            if pad is not None:
                buttons, rt_val = pad
                y_now = bool(buttons & XINPUT_BUTTON_Y)
                rt_now = rt_val > RT_THRESHOLD

                # 边缘检测: 只在从"未按"变成"按下"的瞬间触发，防止长按连续触发
                if y_now and not _prev_y_pressed:
                    attack_name = "Y (踏步斩)"
                    attack_triggered = True
                if rt_now and not _prev_rt_pressed:
                    attack_name = "RT (气刃斩)"
                    attack_triggered = True

                _prev_y_pressed = y_now
                _prev_rt_pressed = rt_now

                # 在画面左上角显示手柄状态
                pad_info = f"Pad: Y={'ON' if y_now else '--'} RT={rt_val:3d}"
                cv2.putText(display_frame, pad_info, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Pad: Not Connected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if attack_triggered:
                # === 帧差法: 存底图 → 等待命中 → 高速采帧与底图做差 → OCR ===
                baseline = cap.get_screenshot()  # 攻击刚按下，数字还没出来，此帧为底图
                print(f"\n>>> 检测到手柄 {attack_name}，已存底图，等待 {DETECT_DELAY}s...")
                time.sleep(DETECT_DELAY)

                t_start = time.time()
                t_end = t_start + DETECT_WINDOW
                fcount = 0
                best_diff_pixels = 0
                best_result = None
                best_frame = None
                all_ocr_readings = []

                while time.time() < t_end:
                    f = cap.get_screenshot()
                    fcount += 1

                    result = processor.detect_damage_diff(baseline, f, roi)
                    dp = result['diff_pixels']

                    # 逐帧日志
                    ocr_str = ""
                    if result['ocr_details']:
                        vals = [str(d['value']) if d['value'] else '?' for d in result['ocr_details']]
                        ocr_str = f" ocr=[{','.join(vals)}]"
                    print(f"      帧{fcount:2d} | 差值像素:{dp:6d} | 轮廓:{len(result['ocr_details'])}{ocr_str}")

                    for d in result['ocr_details']:
                        all_ocr_readings.append({**d, 'frame': fcount})

                    if dp > best_diff_pixels:
                        best_diff_pixels = dp
                        best_result = result
                        best_frame = f

                elapsed = time.time() - t_start
                detected = any(r['value'] is not None for r in all_ocr_readings)
                tag = "HIT" if detected else "MISS"

                print(f"    ----")
                print(f"    结果: {tag} | 差值像素峰值: {best_diff_pixels}")
                print(f"    采帧:{fcount} 耗时:{elapsed:.3f}s FPS:{fcount/elapsed:.1f}")

                if all_ocr_readings:
                    ok = [r for r in all_ocr_readings if r['value'] is not None]
                    fail = [r for r in all_ocr_readings if r['value'] is None]
                    print(f"    ---- OCR 报告 ({len(all_ocr_readings)} 次) 成功:{len(ok)} 失败:{len(fail)} ----")
                    for i, r in enumerate(all_ocr_readings):
                        status = f"= {r['value']}" if r['value'] is not None else "= FAIL"
                        print(f"      [{i+1}] 帧{r['frame']} | OCR{status:>8s} | "
                              f"原始:\"{r['raw_text']}\" | bbox:{r['bbox']} area:{r['area']}")
                else:
                    if best_diff_pixels == 0:
                        print(f"    [诊断] 帧差为 0 → 画面无变化 (可能没打中/怪太远)")
                    elif best_diff_pixels < 500:
                        print(f"    [诊断] 帧差很小 ({best_diff_pixels}px) → 仅有轻微晃动，无伤害数字")
                    else:
                        print(f"    [诊断] 有 {best_diff_pixels} 差值像素但无轮廓通过过滤")

                timestamp = time.strftime("%H:%M:%S")
                ocr_nums = [str(r['value']) for r in all_ocr_readings if r['value'] is not None]
                nums_str = ",".join(ocr_nums) if ocr_nums else "-"
                damage_history.append(
                    f"[{timestamp}] {tag} {attack_name} ocr=[{nums_str}] diff={best_diff_pixels} frm={fcount}"
                )
                if len(damage_history) > 8:
                    damage_history.pop(0)

                # 显示帧差 mask
                if best_result is not None:
                    cv2.imshow("Diff Mask (white=changed)", best_result['diff_mask'])

                # 显示 ROI 区域对比 (底图 vs 最佳帧)
                if best_frame is not None:
                    base_crop = baseline[y:y+h, x:x+w]
                    best_crop = best_frame[y:y+h, x:x+w]
                    disp_h = min(350, base_crop.shape[0])
                    sc = disp_h / base_crop.shape[0]
                    base_small = cv2.resize(base_crop, None, fx=sc, fy=sc)
                    best_small = cv2.resize(best_crop, None, fx=sc, fy=sc)
                    # 在最佳帧上标注 OCR 结果
                    if best_result and best_result['ocr_details']:
                        for d in best_result['ocr_details']:
                            bx2, by2, bw2, bh2 = d['bbox']
                            cv2.rectangle(best_small, (int(bx2*sc), int(by2*sc)),
                                          (int((bx2+bw2)*sc), int((by2+bh2)*sc)), (0, 255, 0), 2)
                            label = str(d['value']) if d['value'] else '?'
                            cv2.putText(best_small, label, (int(bx2*sc), int(by2*sc)-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    compare = np.hstack([base_small, best_small])
                    cv2.putText(compare, "Baseline", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    cv2.putText(compare, "Best Frame", (base_small.shape[1]+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.imshow("Baseline vs Best (side by side)", compare)

                # OCR 裁剪图拼接
                if best_result and best_result['ocr_details']:
                    thresh_imgs = []
                    for d in best_result['ocr_details']:
                        img = d['thresh_img']
                        sh = 60 / max(img.shape[0], 1)
                        resized = cv2.resize(img, None, fx=sh, fy=sh, interpolation=cv2.INTER_NEAREST)
                        label = str(d['value']) if d['value'] else '?'
                        labeled = cv2.copyMakeBorder(resized, 20, 0, 0, 5, cv2.BORDER_CONSTANT, value=0)
                        cv2.putText(labeled, label, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                        thresh_imgs.append(labeled)
                    if thresh_imgs:
                        max_h = max(im.shape[0] for im in thresh_imgs)
                        padded = [cv2.copyMakeBorder(im, 0, max_h-im.shape[0], 0, 0,
                                                     cv2.BORDER_CONSTANT, value=0)
                                  if im.shape[0] < max_h else im for im in thresh_imgs]
                        cv2.imshow("OCR Crops (thresh)", np.hstack(padded))
            else:
                # 未触发攻击时，显示等待提示
                cv2.putText(display_frame, "Waiting... Press Y/RT on controller to attack",
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示历史列表 (在 ROI 下方)
            hist_y = y + h + 20
            for i, record in enumerate(reversed(damage_history)):
                cv2.putText(display_frame, record, (x, hist_y + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        elif mode == '4':
            if is_active:
                color, ratio, mask = processor.analyze_color_state(current_frame, roi, processor.SHARPNESS_COLORS)
                cv2.putText(display_frame, f"Frame: {color} ({ratio:.1%})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                print(f"\r外框颜色: {color} | 完整度: {ratio:.1%}   ", end="")
                cv2.imshow("Color Mask", mask)
        elif mode == '5':
            if is_active:
                # 颜色判定
                color, _, mask = processor.analyze_color_state(current_frame, roi, processor.SPIRIT_COLORS)
                # 白线气量判定
                line_ratio, level_mask = processor.extract_gauge_level(current_frame, roi, color)

                cv2.putText(display_frame, f"Color: {color} | Level: {line_ratio:.1%}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # 在画面上画出识别到的线的位置
                line_x = int(x + w * line_ratio)
                cv2.line(display_frame, (line_x, y), (line_x, y+h), (0, 255, 0), 2)

                # [红刃调试] 在画面上叠加亮度剖面曲线
                if color in ['red', 'red_2']:
                    crop_dbg = current_frame[y:y+h, x:x+w]
                    hsv_dbg = cv2.cvtColor(crop_dbg, cv2.COLOR_BGR2HSV)
                    v_dbg = hsv_dbg[:, :, 2].astype(np.float32)
                    col_means = np.mean(v_dbg, axis=0)
                    # 重度平滑 (与 vision.py 同参数)
                    ks = max(3, w // 6)
                    if ks % 2 == 0: ks += 1
                    smoothed = cv2.GaussianBlur(col_means.reshape(1, -1), (ks, 1), 0).flatten()
                    # peak/valley 中点阈值 (与 vision.py 同逻辑)
                    peak_v = np.max(smoothed)
                    valley_v = np.min(smoothed)
                    contrast_v = peak_v - valley_v
                    thresh_val = (peak_v + valley_v) / 2
                    # 绘制亮度曲线 (黄色，ROI 上方)
                    profile_h = 60
                    profile_top = max(0, y - profile_h - 25)
                    max_v = max(peak_v, 1)
                    pts = []
                    for px in range(w):
                        py = int((1.0 - smoothed[px] / max_v) * profile_h)
                        pts.append((x + px, profile_top + py))
                    pts_arr = np.array(pts, dtype=np.int32)
                    cv2.polylines(display_frame, [pts_arr], False, (0, 255, 255), 2)
                    # 阈值线 (青色)
                    if max_v > 0:
                        thresh_y = profile_top + int((1.0 - thresh_val / max_v) * profile_h)
                        cv2.line(display_frame, (x, thresh_y), (x + w, thresh_y), (255, 255, 0), 1)
                    cv2.putText(display_frame, f"Pk:{peak_v:.0f} Vl:{valley_v:.0f} C:{contrast_v:.0f} Thr:{thresh_val:.0f}",
                                (x, profile_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                print(f"\r内槽颜色: {color} | 气量(白线): {line_ratio:.1%}   ", end="")
                cv2.imshow("Color Mask", level_mask) # 显示气量计算用的 Mask
        else:
            # 智能调整文字位置，防止超出屏幕上边缘
            text_y = y - 10 if y > 30 else y + h + 30

            if is_active:
                val, thresh = processor.extract_digits(current_frame, roi)
                display_val = str(val) if val is not None else "N/A"

                # --- OCR 历史记录 ---
                if val is not None:
                    should_record = False
                    if not damage_history:
                        should_record = True
                    else:
                        # 简单去重: 数值改变 或 时间间隔 > 1s
                        last_val_str = damage_history[-1].split("] ")[-1]
                        if str(val) != last_val_str or (time.time() - last_record_time > 1.0):
                            should_record = True
                    
                    if should_record:
                        timestamp = time.strftime("%H:%M:%S")
                        damage_history.append(f"[{timestamp}] {val}")
                        if len(damage_history) > 8:
                            damage_history.pop(0)
                        last_record_time = time.time()
                
                # 绘制绿色文字 (字体放大一倍，加粗)
                cv2.putText(display_frame, f"OCR: {display_val}", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("OCR Debug (Threshold)", thresh)
            
            
            # 显示历史列表
            hist_y = text_y + 30
            for i, record in enumerate(reversed(damage_history)):
                cv2.putText(display_frame, record, (x, hist_y + i*20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Verification View", display_frame)
        mouse_frame = current_frame # 更新全局帧供取色使用 (使用无框的原始帧)

        if cv2.waitKey(1) & 0xFF == ord(';'):
            break
            
    keyboard.unhook_all()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate()