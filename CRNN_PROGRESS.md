# CRNN 伤害数字识别 — 进度追踪

## 项目目标
用轻量级 CRNN（CNN + BiLSTM + CTC）替代 Tesseract OCR，提升伤害数字识别的速度和准确率。

---

## 实施进度

| 步骤 | 描述 | 状态 | 备注 |
|------|------|------|------|
| 1 | CRNN 模型定义 (`models/crnn/architecture.py`) | ✅ 完成 | ~1.4M 参数，前向传播已验证 |
| 2 | 推理器 (`models/crnn/recognizer.py`) | ✅ 完成 | 支持单张/批量推理 + CTC 解码 |
| 3 | 合成数据生成器 (`tools/generate_synthetic.py`) | ✅ 完成 | 默认生成 50,000 张 |
| 4 | 训练脚本 (`tools/train_crnn.py`) | ✅ 完成 | Phase A/B，TensorBoard 日志 |
| 5 | 合成数据预训练 (Phase A) | ⬜ 待执行 | 目标: 验证集准确率 >95% |
| 6 | 游戏数据自动采集 (`tools/collect_data.py`) | ✅ 完成 | XInput 手柄触发 |
| 7 | 标注校正工具 (`tools/label_editor.py`) | ✅ 完成 | OpenCV UI，键盘输入 |
| 8 | 真实数据微调 (Phase B) | ⬜ 待执行 | 真实30% + 合成70% |
| 9 | 评估脚本 (`tools/evaluate_crnn.py`) | ✅ 完成 | CRNN vs Tesseract 对比 |
| 10 | 集成到 `utils/vision.py` | ✅ 完成 | 批量 CRNN 推理 + Tesseract 回退 |
| 11 | 集成到 `envs/game_env.py` | ✅ 完成 | 传入 crnn_model_path |
| 12 | 更新 `requirements.txt` | ✅ 完成 | 添加 Pillow，pytesseract 改为可选 |
| 13 | 速度基准测试 (`tools/benchmark_speed.py`) | ✅ 完成 | p50/p95/p99 延迟 |
| 14 | 端到端验证 | ⬜ 待执行 | calibrate_roi.py 模式3 实时测试 |

---

## 下一步操作（按顺序）

### Phase A：合成数据预训练
```bash
# 1. 生成合成数据
python tools/generate_synthetic.py --num 50000

# 2. 预训练（约 100 epochs）
python tools/train_crnn.py --phase A --epochs 100

# 3. 验证准确率
python tools/evaluate_crnn.py --syn-val

# 4. 速度测试
python tools/benchmark_speed.py
```

### Phase B：真实数据微调（需要游戏运行）
```bash
# 5. 游戏中采集数据（按手柄 Y/RT 攻击自动采集）
python tools/collect_data.py

# 6. 人工校正标签（目标 500-1000 张）
python tools/label_editor.py

# 7. 微调训练
python tools/train_crnn.py --phase B --train-dir data/raw_captures --resume models/crnn/damage_crnn_best.pth --epochs 50

# 8. 对比评估
python tools/evaluate_crnn.py --data-dir data/raw_captures
```

### 端到端验证
```bash
# 9. 实时测试（calibrate_roi.py 模式3）
python tools/calibrate_roi.py

# 10. 回退测试：删除模型文件后确认自动使用 Tesseract
```

---

## 文件清单

### 新建文件
| 文件 | 说明 |
|------|------|
| `models/crnn/__init__.py` | 包初始化 |
| `models/crnn/architecture.py` | CRNN 模型 (CNN + BiLSTM + CTC) |
| `models/crnn/recognizer.py` | 推理封装 (预处理、批量推理、CTC 解码) |
| `tools/generate_synthetic.py` | 合成数据生成 (PIL 渲染 + 增强) |
| `tools/train_crnn.py` | 训练脚本 (CTC Loss, Adam, 检查点) |
| `tools/collect_data.py` | 游戏中自动采集伤害截图 |
| `tools/label_editor.py` | 标注校正 UI |
| `tools/evaluate_crnn.py` | CRNN vs Tesseract 准确率对比 |
| `tools/benchmark_speed.py` | 速度基准测试 |

### 修改文件
| 文件 | 改动 |
|------|------|
| `utils/vision.py` | `__init__` 增加 `crnn_model_path`；`detect_hit_signals` 改为先收集候选再批量 CRNN 推理；无模型时回退 Tesseract |
| `envs/game_env.py` | 第30行传入 `crnn_model_path="models/crnn/damage_crnn_best.pth"` |
| `requirements.txt` | 添加 `Pillow`，`pytesseract` 改为注释(可选) |

---

## 验证标准
1. 合成验证集序列准确率 >95%
2. GPU 推理延迟 <5ms/crop
3. calibrate_roi.py 模式3 实时效果无退化
4. 删除模型文件后自动回退到 Tesseract
