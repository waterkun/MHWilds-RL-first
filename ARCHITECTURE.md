# MHWilds-RL 系统架构文档

本文档详细描述了《怪物猎人：荒野》强化学习智能体的系统架构、模块划分及数据流向。

## 1. 系统概览 (System Overview)

本系统采用经典的 **Agent-Environment** 交互模式。由于游戏本身没有提供 API，我们需要构建一个“代理环境”层，将屏幕画面转化为状态（State），将神经网络输出转化为手柄操作（Action）。

```mermaid
graph TD
    subgraph "Environment (Host Machine)"
        Game[Monster Hunter Wilds]
        Display[Screen Output]
        Input[Virtual Controller Driver]
    end

    subgraph "RL System (Python)"
        Capture[Screen Capture Module]
        Processor[Image Preprocessing & OCR]
        Agent[RL Agent (PPO/SAC)]
        Wrapper[Gym Environment Wrapper]
        Controller[Action Executor]
    end

    Game --> Display
    Display --> Capture
    Capture --> Processor
    Processor --> Wrapper
    Wrapper -->|State (Obs)| Agent
    Agent -->|Action Index| Wrapper
    Wrapper -->|Button Signal| Controller
    Controller --> Input
    Input --> Game
```

## 2. 核心模块详解

### 2.1 感知层 (Perception Layer)
负责从游戏中提取信息，构建智能体的观察空间（Observation Space）。

- **屏幕捕获 (Screen Capture)**:
  - 使用 `mss` 或 `d3dshot` 进行高频屏幕截图（目标 > 60 FPS）。
  - 仅捕获游戏窗口区域，支持动态分辨率缩放。

- **特征提取 (Feature Extraction)**:
  - **视觉特征**: 将原始 RGB 图像转换为灰度图，调整大小为 `84x84` 或 `128x128`，并进行帧堆叠（Frame Stacking, 通常 4 帧）以捕捉动作的运动轨迹。
  - **数值特征 (UI Parsing)**:
    - **血量/耐力**: 使用 OpenCV 进行颜色阈值分割或模板匹配，计算血条/耐力条的像素占比。
    - **斩味/弹药**: 识别当前武器状态。
    - **伤害数字**: (可选) 使用 OCR 识别跳出的伤害数字作为奖励信号的一部分。

### 2.2 决策层 (Decision Layer)
智能体的大脑，基于当前状态输出动作概率分布。

- **算法**: 推荐使用 **PPO (Proximal Policy Optimization)** 或 **SAC (Soft Actor-Critic)**。
- **网络结构 (Policy Network)**:
  - **CNN Backbone**: 类似于 NatureCNN 或 ResNet，用于处理视觉输入。
  - **MLP Head**: 处理数值特征（如血量百分比），与 CNN 的输出特征拼接（Concatenate）。
  - **LSTM (可选)**: 如果需要处理长时记忆（如记住怪物的换区行为），可引入循环神经网络层。

### 2.3 执行层 (Action Layer)
将神经网络的抽象输出映射为具体的游戏操作。

- **动作空间 (Action Space)**:
  - 采用 `MultiDiscrete` 空间，因为手柄允许同时按下多个键（如 `R2` + `Triangle`）。
  - 或者定义高层宏动作（Macro Actions），例如：`[移动, 翻滚, 攻击1, 攻击2, 喝药]`。
- **虚拟手柄**:
  - 使用 `vgamepad` 库模拟 Xbox 360 手柄信号。
  - 实现 `Hold` 和 `Press` 的逻辑区分（例如蓄力攻击需要长按）。

### 2.4 环境封装 (Gym Wrapper)
遵循 OpenAI Gymnasium 接口标准，实现自定义环境类 `MHWildsEnv`。

- **Reset**: 重置游戏状态（如加载存档、返回营地）。
- **Step**: 执行动作 -> 等待 N 帧（Frame Skip） -> 获取新一帧 -> 计算奖励 -> 判断是否结束。
- **Reward Function (奖励函数)**:
  - $R_{total} = w_1 \cdot R_{damage} + w_2 \cdot R_{survival} + w_3 \cdot R_{position}$
  - **生存奖励**: 存活时间越长奖励越高，掉血扣分。
  - **攻击奖励**: 造成伤害（通过视觉识别或内存钩子）给予正反馈。

## 3. 数据流与时序 (Data Flow)

1. **T0**: `Env.step(action)` 被调用。
2. **T0 -> T1**: `Action Executor` 解析动作，向虚拟手柄发送指令（如按下 'Y' 键）。
3. **T1 -> T2**: 游戏引擎处理输入，渲染下一帧画面。
4. **T2**: `Screen Capture` 捕获当前帧。
5. **T2 -> T3**: `Processor` 处理图像，计算当前血量变化，生成 `Next_Observation`。
6. **T3**: 计算 `Reward`（例如：如果血量减少，Reward -= 10）。
7. **T4**: 返回 `(obs, reward, done, info)` 给 RL 训练循环。

## 4. 技术栈选型 (Tech Stack)

- **编程语言**: Python 3.9+
- **深度学习框架**: PyTorch
- **RL 库**: Stable-Baselines3
- **图像处理**: OpenCV, NumPy
- **输入模拟**: ViGEmBus (驱动), vgamepad (Python库)
- **OCR**: Tesseract / EasyOCR