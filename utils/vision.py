import cv2
import numpy as np
from collections import deque
import os

try:
    import pytesseract
    # 指定 Tesseract-OCR 的安装路径
    # 如果你安装在其他盘符，请修改下面的路径
    pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'
except ImportError:
    pytesseract = None

class ImageProcessor:
    def __init__(self, input_shape=(84, 84), crnn_model_path=None):
        """
        :param input_shape: 神经网络期望的输入尺寸 (width, height)
        :param crnn_model_path: CRNN 模型文件路径，None 时回退到 Tesseract
        """
        self.input_shape = input_shape
        self.crnn = None

        if crnn_model_path and os.path.exists(crnn_model_path):
            try:
                from models.crnn.recognizer import CRNNRecognizer
                self.crnn = CRNNRecognizer(crnn_model_path)
                print(f"[ImageProcessor] CRNN loaded: {crnn_model_path}")
            except Exception as e:
                print(f"[ImageProcessor] CRNN load failed ({e}), falling back to Tesseract")
                self.crnn = None
        elif crnn_model_path:
            print(f"[ImageProcessor] CRNN model not found: {crnn_model_path}, using Tesseract")

    def preprocess_frame(self, frame):
        """
        预处理画面供 Agent 使用：
        1. 灰度化
        2. 缩放
        3. 归一化 (可选，通常 Gym Wrapper 会处理，这里先返回 uint8)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 缩放到目标尺寸 (例如 84x84)
        resized = cv2.resize(gray, self.input_shape, interpolation=cv2.INTER_AREA)
        
        # 增加一个维度以匹配 (H, W, C) 格式，虽然是单通道
        processed = resized[:, :, np.newaxis]
        
        return processed

    def extract_health_bar(self, frame, roi):
        """
        从画面中提取血量百分比 (基于颜色阈值)
        :param frame: 原始 BGR 图像
        :param roi: 感兴趣区域 (x, y, w, h) - 需要根据游戏UI手动测量
        :return: float (0.0 - 1.0)
        """
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]
        
        # 转换为 HSV 空间以便进行颜色分割
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # 定义绿色的范围 (怪物猎人血条通常是绿色)
        # 注意：这些值需要根据实际游戏截图进行校准
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        # 创建掩码
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 计算非零像素点 (即绿色像素)
        green_pixels = cv2.countNonZero(mask)
        total_pixels = w * h # 或者 mask.size
        
        # 简单的占比计算
        health_ratio = green_pixels / total_pixels if total_pixels > 0 else 0.0
        
        return health_ratio, mask # 返回 mask 用于调试

    def extract_digits(self, frame, roi):
        """
        使用 OCR 识别指定区域的数字 (例如伤害数值、道具数量)
        :param frame: 原始 BGR 图像
        :param roi: (x, y, w, h)
        :return: int (识别到的数字) 或 None (未识别到)
        """
        x, y, w, h = roi
        # 边界检查
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
            
        crop = frame[y:y+h, x:x+w]
        
        # 图像预处理：灰度 -> 二值化 (黑白分明利于 OCR)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # 使用 Otsu 自动阈值，或者根据游戏字体颜色手动调整
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if pytesseract is None:
            return None, thresh

        # 配置 Tesseract:
        # --psm 7: 将图像视为单行文本
        # outputbase digits: 限制只识别数字
        config = r'--psm 7 -c tessedit_char_whitelist=0123456789'
        
        try:
            text = pytesseract.image_to_string(thresh, config=config)
            # 过滤掉非数字字符（以防万一）
            digits = ''.join(filter(str.isdigit, text))
            return int(digits) if digits else None, thresh
        except Exception:
            return None, thresh

    def detect_hit_signals(self, frame, roi, max_ocr=5, debug=False):
        """
        检测屏幕上的伤害数字 (颜色 + 形状过滤 + OCR 验证)
        流程: 颜色检测 → 轮廓形状过滤 → OCR 读取数字 → 返回实际伤害值
        角色闪光/环境高光无法被 OCR 识别为数字，从而被排除
        :param frame: 原始图像
        :param roi: 搜索区域 (x, y, w, h)
        :param max_ocr: 最大 OCR 调用次数 (限制性能开销)
        :param debug: 如果为 True，额外返回每次 OCR 的详细结果列表
        :return: (total_damage, hit_count, raw_mask, filtered_mask)
                 当 debug=True 时返回第5个元素:
                 ocr_details: list of dict, 每个包含:
                   'value': OCR 识别到的数字 (int) 或 None (识别失败)
                   'raw_text': Tesseract 原始返回文本
                   'bbox': (bx, by, bw, bh) 在 ROI 内的轮廓边界框
                   'thresh_img': 送入 Tesseract 的二值化图像
                   'area': 轮廓面积
                   'aspect': 长宽比
        """
        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]

        # 转换到 HSV 和灰度
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # --- 定义伤害数字的颜色范围 ---
        # 1. 黄色/橙色 (弱点/暴击)
        lower_yellow = np.array([15, 60, 180])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 2. 白色 (普通伤害) - 收紧范围
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # 合并掩码
        mask = cv2.bitwise_or(mask_yellow, mask_white)

        # 膨胀操作：把断裂的数字笔画连成一个整体 (reduced to prevent merging)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # --- 轮廓形状过滤 (预筛选，减少 OCR 调用) ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_mask = np.zeros_like(mask)
        total_damage = 0
        hit_count = 0
        ocr_calls = 0
        candidates = []  # 收集通过形状过滤的候选区域，最后批量识别
        ocr_details = [] if debug else None
        rejected_stats = {'area_small': 0, 'area_big': 0, 'aspect': 0,
                          'solidity': 0, 'extent': 0, 'sizes': []} if debug else None

        # 面积上限根据 ROI 大小自适应
        # 原来固定 5000 对小 ROI 合理，但大 ROI (如 1285x1232) 伤害数字面积会更大
        roi_area = w * h
        area_max = max(5000, int(roi_area * 0.05))  # 最多占 ROI 5%
        area_min = max(30, int(roi_area * 0.00003))  # 最少占 ROI 0.003%

        # 按面积降序排列，优先处理最大的轮廓 (最可能是伤害数字)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            if ocr_calls >= max_ocr:
                break

            area = cv2.contourArea(cnt)
            # 面积过滤
            if area < area_min:
                if debug: rejected_stats['area_small'] += 1
                continue
            if area > area_max:
                if debug:
                    rejected_stats['area_big'] += 1
                    rejected_stats['sizes'].append(int(area))
                continue

            # 长宽比过滤
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect = bw / bh
            if aspect < 0.15 or aspect > 6.0:
                if debug: rejected_stats['aspect'] += 1
                continue

            # 密实度 (Solidity)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
                if solidity < 0.2:
                    if debug: rejected_stats['solidity'] += 1
                    continue

            # 填充度 (Extent)
            extent = area / (bw * bh)
            if extent < 0.15:
                if debug: rejected_stats['extent'] += 1
                continue

            # --- 形状过滤通过 → 收集裁切图 ---
            pad = 5
            ocr_x1 = max(0, bx - pad)
            ocr_y1 = max(0, by - pad)
            ocr_x2 = min(w, bx + bw + pad)
            ocr_y2 = min(h, by + bh + pad)

            # 最小尺寸检查
            if (ocr_x2 - ocr_x1) < 8 or (ocr_y2 - ocr_y1) < 8:
                continue

            digit_crop = gray[ocr_y1:ocr_y2, ocr_x1:ocr_x2]
            ocr_calls += 1

            # 收集候选区域信息
            candidates.append({
                'crop': digit_crop,
                'cnt': cnt,
                'bbox': (bx, by, bw, bh),
                'area': area,
                'aspect': round(aspect, 2),
            })

        # --- 批量识别 (CRNN 或 Tesseract) ---
        if self.crnn is not None and candidates:
            # CRNN 批量推理
            crop_images = [c['crop'] for c in candidates]
            results = self.crnn.recognize_batch(crop_images)

            for i, (value, confidence) in enumerate(results):
                c = candidates[i]
                if value is not None:
                    total_damage += value
                    hit_count += 1
                    cv2.drawContours(filtered_mask, [c['cnt']], -1, 255, -1)

                if debug:
                    ocr_details.append({
                        'value': value,
                        'raw_text': str(value) if value else '',
                        'bbox': c['bbox'],
                        'thresh_img': c['crop'].copy(),
                        'area': c['area'],
                        'aspect': c['aspect'],
                    })

        elif pytesseract is not None and candidates:
            # Tesseract 逐个推理 (回退路径)
            config = r'--psm 7 -c tessedit_char_whitelist=0123456789'
            for c in candidates:
                digit_crop = c['crop']
                # 放大图像以提高 OCR 准确率
                scale = 3
                digit_crop = cv2.resize(digit_crop, None, fx=scale, fy=scale,
                                        interpolation=cv2.INTER_CUBIC)
                _, thresh = cv2.threshold(digit_crop, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.bitwise_not(thresh)

                try:
                    text = pytesseract.image_to_string(thresh, config=config)
                    digits = ''.join(filter(str.isdigit, text))
                    if digits:
                        total_damage += int(digits)
                        hit_count += 1
                        cv2.drawContours(filtered_mask, [c['cnt']], -1, 255, -1)

                    if debug:
                        ocr_details.append({
                            'value': int(digits) if digits else None,
                            'raw_text': text.strip(),
                            'bbox': c['bbox'],
                            'thresh_img': thresh.copy(),
                            'area': c['area'],
                            'aspect': c['aspect'],
                        })
                except Exception:
                    if debug:
                        ocr_details.append({
                            'value': None,
                            'raw_text': '[ERROR]',
                            'bbox': c['bbox'],
                            'thresh_img': thresh.copy(),
                            'area': c['area'],
                            'aspect': c['aspect'],
                        })

        if debug:
            return total_damage, hit_count, mask, filtered_mask, ocr_details, rejected_stats
        return total_damage, hit_count, mask, filtered_mask

    def direct_ocr_scan(self, frame, roi):
        """
        直接对 ROI 区域运行 Tesseract 寻找数字 (跳过颜色/形状管线)
        适用于大 ROI 或颜色管线失效时的 OCR 准确率测试
        使用 psm 11 (稀疏文本) 在整个区域中搜索数字
        :param frame: 原始 BGR 图像
        :param roi: (x, y, w, h)
        :return: dict {
            'numbers': list of {'text': str, 'x': int, 'y': int, 'w': int, 'h': int, 'conf': float},
            'raw_text': str,
            'thresh': 处理后的图像 (调试用),
            'elapsed': float (耗时秒)
        }
        """
        if pytesseract is None:
            return {'numbers': [], 'raw_text': '', 'thresh': None, 'elapsed': 0}

        import time as _time
        t0 = _time.time()

        x, y, w, h = roi
        crop = frame[y:y+h, x:x+w]

        # 缩小到合理尺寸以加速 OCR (最大宽度 800px)
        max_w = 800
        scale = 1.0
        if w > max_w:
            scale = max_w / w
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 自适应阈值: 伤害数字通常是亮色 (白/黄) 的文字
        # 使用 Otsu 二值化后反转 (Tesseract 偏好白底黑字)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        # psm 11 = 稀疏文本，在整个图像中寻找散落的文字
        # psm 12 = 同上但带 OSD
        config = r'--psm 11 -c tessedit_char_whitelist=0123456789'

        numbers = []
        raw_text = ''
        try:
            # 获取带位置的详细结果
            data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
            raw_text = ' '.join(t for t in data['text'] if t.strip())

            for i, txt in enumerate(data['text']):
                digits = ''.join(filter(str.isdigit, txt))
                if digits and data['conf'][i] > 0:
                    # 还原到原始 ROI 坐标
                    nx = int(data['left'][i] / scale)
                    ny = int(data['top'][i] / scale)
                    nw = int(data['width'][i] / scale)
                    nh = int(data['height'][i] / scale)
                    numbers.append({
                        'text': digits,
                        'x': nx, 'y': ny, 'w': nw, 'h': nh,
                        'conf': data['conf'][i],
                    })
        except Exception as e:
            raw_text = f'[ERROR: {e}]'

        elapsed = _time.time() - t0
        return {'numbers': numbers, 'raw_text': raw_text, 'thresh': thresh, 'elapsed': elapsed}

    def detect_damage_diff(self, baseline_frame, current_frame, roi,
                           diff_thresh=40, min_area=80, max_ocr=5):
        """
        帧差法检测伤害数字:
        1. 计算 current - baseline 的绝对差值
        2. 差值大的区域 = 新出现的东西 (伤害数字、特效)
        3. 对差值区域找轮廓 → OCR 验证
        背景像素在两帧中几乎相同，差值≈0，自动消除

        :param baseline_frame: 攻击前的"干净"帧 (BGR)
        :param current_frame: 攻击后的帧 (BGR)
        :param roi: (x, y, w, h)
        :param diff_thresh: 差值阈值 (0-255)，越大越严格
        :param min_area: 最小轮廓面积
        :param max_ocr: 最大 OCR 调用次数
        :return: dict {
            'total_damage': int,
            'hit_count': int,
            'diff_mask': 差值掩码图像,
            'ocr_details': list of {'value', 'raw_text', 'bbox', 'thresh_img'},
            'diff_pixels': 差值像素总数,
        }
        """
        x, y, w, h = roi
        base_crop = baseline_frame[y:y+h, x:x+w]
        curr_crop = current_frame[y:y+h, x:x+w]

        # 灰度差值
        base_gray = cv2.cvtColor(base_crop, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(curr_gray, base_gray)

        # 阈值化: 只保留变化显著的区域
        _, diff_mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

        # 轻度膨胀: 连接断裂的数字笔画 (比原方案少，因为背景已消除)
        kernel = np.ones((2, 2), np.uint8)
        diff_mask = cv2.dilate(diff_mask, kernel, iterations=1)

        diff_pixels = cv2.countNonZero(diff_mask)

        # 找轮廓
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        total_damage = 0
        hit_count = 0
        ocr_details = []
        ocr_calls = 0

        for cnt in contours:
            if ocr_calls >= max_ocr:
                break

            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh == 0:
                continue
            aspect = bw / bh
            # 伤害数字通常是横向的 (宽 > 高)，过滤极端比例
            if aspect < 0.3 or aspect > 8.0:
                continue

            # OCR
            pad = 5
            ox1 = max(0, bx - pad)
            oy1 = max(0, by - pad)
            ox2 = min(w, bx + bw + pad)
            oy2 = min(h, by + bh + pad)
            if (ox2 - ox1) < 8 or (oy2 - oy1) < 8:
                continue

            digit_crop = curr_gray[oy1:oy2, ox1:ox2]
            # 放大
            scale = max(1, 3 if bh < 30 else 2)
            digit_crop = cv2.resize(digit_crop, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(digit_crop, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.bitwise_not(thresh)

            if pytesseract is None:
                continue

            config = r'--psm 7 -c tessedit_char_whitelist=0123456789'
            ocr_calls += 1
            raw_text = ''
            value = None
            try:
                text = pytesseract.image_to_string(thresh, config=config)
                raw_text = text.strip()
                digits = ''.join(filter(str.isdigit, text))
                if digits:
                    value = int(digits)
                    total_damage += value
                    hit_count += 1
            except Exception:
                raw_text = '[ERROR]'

            ocr_details.append({
                'value': value,
                'raw_text': raw_text,
                'bbox': (bx, by, bw, bh),
                'thresh_img': thresh.copy(),
                'area': int(area),
            })

        return {
            'total_damage': total_damage,
            'hit_count': hit_count,
            'diff_mask': diff_mask,
            'ocr_details': ocr_details,
            'diff_pixels': diff_pixels,
        }

    def analyze_color_state(self, frame, roi, color_map, threshold=0.01):
        """
        统计区域内各颜色的像素数量，返回占比最高的颜色 (用于识别斩位/气刃等级)
        :param frame: 原始图像
        :param roi: (x, y, w, h)
        :param color_map: dict { 'name': (lower_np, upper_np) }
        :param threshold: 最小像素占比阈值 (0.0 - 1.0)，低于此值视为 'none'
        :return: (dominant_color_name, fill_ratio, best_mask)
        """
        x, y, w, h = roi
        if w <= 0 or h <= 0: return 'none', 0.0, None
        
        crop = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # --- 1. 计算填充率 (基于亮度) ---
        # 只要像素足够亮 (V > 100)，就认为是气槽的一部分 (包含红色、黄色、白色波浪)
        # 这相当于 "Total - Empty(Dark)"
        bright_mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 255, 255]))
        bright_pixels = cv2.countNonZero(bright_mask)
        total_pixels = w * h
        fill_ratio = bright_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # --- 2. 判定颜色状态 ---
        max_pixels = 0
        dominant_color = 'none'
        # 默认显示亮度掩码，方便调试气量
        best_mask = bright_mask 
        
        for name, (lower, upper) in color_map.items():
            mask = cv2.inRange(hsv, lower, upper)
            count = cv2.countNonZero(mask)
            if count > max_pixels:
                max_pixels = count
                dominant_color = name
                # 如果需要调试特定颜色，可以在这里覆盖 best_mask
                # best_mask = mask 
        
        # 如果占比低于阈值（说明可能是背景杂色），强制返回 none
        if fill_ratio < threshold:
            return 'none', 0.0, best_mask
            
        return dominant_color, fill_ratio, best_mask

    def extract_gauge_level(self, frame, roi, gauge_color='none'):
        """
        根据气刃状态选择不同的气量提取算法
        :return: (ratio, debug_mask)
        """
        x, y, w, h = roi
        if w <= 0 or h <= 0: return 0.0, None
        
        crop = frame[y:y+h, x:x+w]
        
        # 策略分支
        if gauge_color in ['red', 'red_2']:
            # 红刃状态：基于列亮度剖面的相对比较法 (抗闪烁干扰)
            return self._extract_level_brightness_profile(crop, w, h)
        else:
            # 非红刃状态：使用原有的垂直线扫描 (已验证非常准确，保持不变)
            return self._extract_level_vertical_scan(crop, w, h)

    def _extract_level_vertical_scan(self, crop, w, h):
        """
        原有的垂直线扫描算法 (保留给非红刃使用)
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # 1. 白色高亮 (S低, V高)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # 2. 金色高亮 (H黄色区间, V高)
        lower_gold = np.array([10, 40, 200])
        upper_gold = np.array([40, 255, 255])
        mask_gold = cv2.inRange(hsv, lower_gold, upper_gold)
        
        mask_line = cv2.bitwise_or(mask_white, mask_gold)
        
        col_sums = np.sum(mask_line, axis=0) / 255 
        threshold_line = h * 0.4
        candidates = np.where(col_sums > threshold_line)[0]
        
        valid_line_x = -1
        
        if len(candidates) > 0:
            candidates = np.sort(candidates)[::-1]
            for c in candidates:
                # 检查右侧暗
                check_w = min(5, w - c - 1)
                if check_w <= 0: continue
                right_strip_v = hsv[:, c+1:c+1+check_w, 2]
                avg_v = np.mean(right_strip_v)
                
                # 检查左侧亮
                check_w_left = min(5, c)
                avg_v_left = 0
                if check_w_left > 0:
                    left_strip_v = hsv[:, c-check_w_left:c, 2]
                    avg_v_left = np.mean(left_strip_v)
                
                if avg_v < 80 and avg_v_left > 80:
                    valid_line_x = c
                    break
        
        if valid_line_x != -1:
            ratio = (valid_line_x + 1) / w
            return min(1.0, max(0.0, ratio)), mask_line
            
        # Fallback: 亮度边缘检测
        mask_bright = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 255, 255]))
        col_sums_bright = np.sum(mask_bright, axis=0) / 255
        bright_cols = np.where(col_sums_bright > h * 0.3)[0]
        
        if len(bright_cols) > 0:
            for i in range(len(bright_cols) - 1, -1, -1):
                col_idx = bright_cols[i]
                if col_idx > 5:
                    left_region = hsv[:, col_idx-5:col_idx, 2]
                    if np.mean(left_region) > 80:
                        ratio = (col_idx + 1) / w
                        return min(1.0, max(0.0, ratio)), mask_bright
            
        return 0.0, mask_line

    def _extract_level_brightness_profile(self, crop, w, h):
        """
        基于列平均亮度的相对比较法 (红刃专用)
        核心思路: 右侧(已消耗)无论是纯暗还是暗红闪烁，亮度始终低于左侧(剩余)
        改进:
        - 用 peak/valley 中点作为阈值 (而非左侧参考的固定比例)，解决近空时跳100%
        - 梯度精定位: 粗扫后在附近找最陡亮度下降点，使绿线对齐实际边缘
        """
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)

        # 1. 每列平均亮度
        col_means = np.mean(v_channel, axis=0)

        # 2. 重度高斯平滑，消除波浪纹和闪烁的局部波动
        kernel_size = max(3, w // 6)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(col_means.reshape(1, -1), (kernel_size, 1), 0).flatten()

        peak = np.max(smoothed)
        valley = np.min(smoothed)
        contrast = peak - valley

        # 整体太暗 → 气量为 0
        if peak < 60:
            return 0.0, self._build_profile_debug(v_channel, smoothed, w, h, 0)

        # 亮度均匀 (无明显亮暗分界) → 整条闪烁 or 满槽
        if contrast < 30:
            if peak > 150:
                return 1.0, self._build_profile_debug(v_channel, smoothed, w, h, w)
            return 0.0, self._build_profile_debug(v_channel, smoothed, w, h, 0)

        # 3. 动态阈值: 亮暗中点 (自动适应各种亮度条件)
        threshold = (peak + valley) / 2

        # 4. 粗定位: 从右向左滑动窗口
        window = max(3, w // 15)
        coarse_boundary = 0
        for x in range(w - window, -1, -1):
            if np.mean(smoothed[x:x + window]) > threshold:
                coarse_boundary = min(x + window, w)
                break

        # 5. 精定位: 在粗定位附近用轻度平滑曲线的梯度找最陡下降点
        light_kernel = max(3, w // 15)
        if light_kernel % 2 == 0:
            light_kernel += 1
        light_smoothed = cv2.GaussianBlur(col_means.reshape(1, -1), (light_kernel, 1), 0).flatten()
        gradient = np.diff(light_smoothed)

        search_radius = max(5, w // 8)
        search_left = max(0, coarse_boundary - search_radius)
        search_right = min(len(gradient), coarse_boundary + search_radius)

        boundary = coarse_boundary
        if search_right > search_left:
            local_grad = gradient[search_left:search_right]
            min_idx = np.argmin(local_grad)
            if local_grad[min_idx] < -3:  # 存在显著亮度下降
                boundary = search_left + min_idx + 1

        boundary = max(0, min(w, boundary))
        ratio = boundary / w
        debug = self._build_profile_debug(v_channel, smoothed, w, h, boundary)
        return min(1.0, max(0.0, ratio)), debug

    def _build_profile_debug(self, v_channel, smoothed, w, h, boundary):
        """
        生成亮度剖面调试图:
        上部 = V 通道原图 (直观看到亮暗分布)
        下部 = 平滑后亮度曲线 (算法实际依据)
        白色竖线 = 检测到的边界位置
        """
        graph_h = max(h, 40)
        debug_h = h + graph_h
        debug = np.zeros((debug_h, w), dtype=np.uint8)

        # 上部: V 通道原图
        debug[:h, :] = v_channel.astype(np.uint8)

        # 下部: 平滑亮度曲线
        max_val = max(np.max(smoothed), 1)
        for x in range(w):
            y_pos = int((1.0 - smoothed[x] / max_val) * (graph_h - 1))
            y_pos = max(0, min(graph_h - 1, y_pos))
            debug[h + y_pos:h + graph_h, x] = 100  # 曲线下方填充
            debug[h + y_pos, x] = 255               # 曲线线条

        # boundary 白色竖线
        if 0 < boundary < w:
            debug[:, min(boundary, w - 1)] = 255

        return debug

    # --- 预定义颜色字典 ---
    # 斩位颜色 (大致范围，需根据实际游戏微调)
    SHARPNESS_COLORS = {
        # [优化] 收紧白色范围：S最大30->20, V最小200->220，防止高亮红被误判
        'white':  (np.array([0, 0, 220]), np.array([180, 20, 255])),
        'blue':   (np.array([100, 150, 150]), np.array([130, 255, 255])),
        'green':  (np.array([40, 50, 50]), np.array([80, 255, 255])),
        # [优化] 提高黄色饱和度门槛 100->150，防止灰色被误判
        'yellow': (np.array([15, 150, 100]), np.array([35, 255, 255])),
        'red':    (np.array([0, 150, 150]), np.array([10, 255, 255])),
        'red_2':  (np.array([170, 150, 150]), np.array([180, 255, 255])), # 新增红色 wrap 范围
    }

    # 太刀气刃槽颜色 (无/白/黄/红)
    # 气刃槽通常是发光的，亮度(V)较高
    SPIRIT_COLORS = {
        # [精准过滤] 提高 S 下限过滤浅红闪烁，降低 V 下限以捕捉深红填充
        'red':    (np.array([0, 150, 50]), np.array([10, 255, 255])),
        'red_2':  (np.array([170, 150, 50]), np.array([180, 255, 255])),
        
        # 黄色：[修复] V 回调到 100，解决非红刃状态下黄色无法识别的问题
        'yellow': (np.array([15, 80, 100]), np.array([35, 255, 255])),
        
        # [移除] 白色：不再单独识别白色，防止波浪特效抢走红色的判定
        # 波浪的白色像素会计入 fill_ratio (因为够亮)，但不会干扰 dominant_color
    }

# 调试代码
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # 创建一个模拟的血条图像
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # 在 (100, 50) 处画一个绿色矩形模拟满血
    cv2.rectangle(dummy_frame, (100, 50), (300, 70), (0, 255, 0), -1) 
    
    # 测试预处理
    obs = processor.preprocess_frame(dummy_frame)
    print(f"Observation shape: {obs.shape}")
    
    # 测试血量提取
    roi = (100, 50, 200, 20) # x, y, w, h
    ratio, mask = processor.extract_health_bar(dummy_frame, roi)
    print(f"Health Ratio: {ratio:.2f}")
    
    cv2.imshow("Processed Observation", obs)
    cv2.imshow("Health Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()