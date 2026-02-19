import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.game_env import MHWildsEnv

def train():
    # 1. 创建目录用于存放模型和日志
    timestamp = int(time.time())
    models_dir = f"models/PPO-{timestamp}"
    log_dir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print(f"模型将保存在: {models_dir}")
    print(f"Tensorboard 日志将保存在: {log_dir}")

    # 2. 初始化环境
    # 使用 DummyVecEnv 包装，SB3 要求向量化环境
    # lambda 函数用于延迟创建环境实例
    env = DummyVecEnv([lambda: MHWildsEnv()])
    
    # 堆叠 4 帧，这样 Agent 能感知运动方向和速度
    # channels_order='last' 对应 (H, W, C)
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # 3. 初始化模型
    # policy="CnnPolicy" 专门用于处理图像输入
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cuda", # 强制使用 GPU
    )

    # 4. 设置回调函数 (每 5000 步保存一次模型)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path=models_dir, 
        name_prefix="mhwilds"
    )

    # 5. 开始训练
    print("🚀 开始训练... (按 Ctrl+C 可以安全停止并保存)")
    try:
        # 训练 100万步 (根据需要调整)
        model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\n⚠️ 检测到中断，正在保存当前模型...")
    finally:
        model.save(f"{models_dir}/mhwilds_final")
        env.close()
        print("✅ 模型已保存，环境已关闭。")

if __name__ == "__main__":
    train()