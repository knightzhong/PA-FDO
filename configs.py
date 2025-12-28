import torch

class Config:
    # 任务设置
    TASK_NAME = 'TFBind8-Exact-v0' # 或 'AntMorphology-Exact-v0'
    
    # 训练参数
    SEED = 42
    BATCH_SIZE = 256
    EPOCHS = 100
    LR = 1e-3
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # SP-RFM 核心参数 (配对策略)
    # 将数据按分数排序，取前 split_ratio 为高分池，后 split_ratio 为低分池
    DATA_SPLIT_RATIO = 0.5 
    # 配对时考虑多少个语义最近邻
    TOP_K_NEIGHBORS = 50
    
    # 模型参数
    LATENT_DIM = 256  # 中间层维度
    
    # 推理参数
    ODE_STEPS = 100   # 欧拉积分步数
    NUM_SAMPLES = 128 # 最终评估生成的样本数