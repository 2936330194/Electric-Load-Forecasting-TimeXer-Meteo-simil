"""
data_factory.py - 数据工厂模块

本模块实现了数据提供器的工厂模式，用于：
1. 根据数据集名称自动选择对应的数据集类
2. 创建数据集实例和数据加载器
3. 统一管理不同数据集的加载接口

设计模式：
    工厂模式（Factory Pattern）：通过 data_dict 字典映射数据集名称到具体类，
    实现了数据集创建的解耦，便于扩展新的数据集类型。

支持的数据集：
    - ETTh1, ETTh2: 小时级 ETT 数据集（电力变压器温度）
    - ETTm1, ETTm2: 分钟级 ETT 数据集
    - custom: 自定义数据集（适用于用户自己的数据）

使用示例：
    data_set, data_loader = data_provider(args, flag='train')
    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
        # 处理批次数据
        pass
"""

# ==================== 导入依赖模块 ====================
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader  # PyTorch 数据加载器


# ==================== 数据集类型映射字典 ====================
# 将数据集名称（字符串）映射到对应的数据集类
# 这是工厂模式的核心：通过键值对实现类型选择
data_dict = {
    "ETTh1": Dataset_ETT_hour,   # ETT 小时级数据集 1（电力变压器温度）
    "ETTh2": Dataset_ETT_hour,   # ETT 小时级数据集 2
    "ETTm1": Dataset_ETT_minute, # ETT 分钟级数据集 1
    "ETTm2": Dataset_ETT_minute, # ETT 分钟级数据集 2
    "custom": Dataset_Custom,    # 自定义数据集，用于加载用户自己的 CSV 数据
}


def data_provider(args, flag):
    """
    数据提供器函数（工厂函数）
    
    根据配置参数创建对应的数据集实例和数据加载器。
    这是整个数据加载流程的入口函数。
    
    参数：
        args: 配置参数对象，需要包含以下属性：
            - data (str): 数据集名称，如 'ETTh1', 'ETTh2', 'custom' 等
            - embed (str): 时间编码类型，'timeF' 表示使用时间特征编码
            - batch_size (int): 批次大小
            - root_path (str): 数据根目录路径
            - data_path (str): 数据文件名（如 'ETTh1.csv'）
            - seq_len (int): 输入序列长度（编码器输入长度）
            - label_len (int): 标签长度（解码器起始部分长度）
            - pred_len (int): 预测长度（解码器预测部分长度）
            - features (str): 特征类型
                             'M' - 多变量预测多变量
                             'S' - 单变量预测单变量
                             'MS' - 多变量预测单变量
            - target (str): 目标变量列名（如 'OT'）
            - freq (str): 时间频率（如 'h' 小时, 't' 分钟）
            - num_workers (int): 数据加载的工作进程数
            
        flag (str): 数据集类型标志
            - 'train': 训练集
            - 'val': 验证集
            - 'test': 测试集
    
    返回：
        data_set: 数据集对象（继承自 torch.utils.data.Dataset）
        data_loader: 数据加载器对象（torch.utils.data.DataLoader）
    
    数据加载器输出：
        每个批次返回四个张量：
        - batch_x: 编码器输入 [batch_size, seq_len, features]
        - batch_y: 解码器目标 [batch_size, label_len + pred_len, features]
        - batch_x_mark: 编码器时间特征 [batch_size, seq_len, time_features]
        - batch_y_mark: 解码器时间特征 [batch_size, label_len + pred_len, time_features]
    """
    # ==================== 获取数据集类 ====================
    # 根据 args.data 从字典中获取对应的数据集类
    Data = data_dict[args.data]
    
    # ==================== 时间编码方式 ====================
    # timeenc=0: 使用固定的时间嵌入（如 PositionalEmbedding）
    # timeenc=1: 使用时间特征编码（TimeFeatureEmbedding），提取更丰富的时间信息
    timeenc = 0 if args.embed != "timeF" else 1

    # ==================== 数据加载器配置 ====================
    # shuffle_flag: 是否打乱数据
    # - 训练集需要打乱以增加随机性
    # - 测试集不打乱以保持时间顺序
    shuffle_flag = False if flag == "test" else True
    
    # drop_last: 是否丢弃最后一个不完整的批次
    # 设置为 False 保留所有数据
    drop_last = False
    
    batch_size = args.batch_size

    # ==================== 创建数据集实例 ====================
    # 调用数据集类的构造函数，传入所有必要参数
    data_set = Data(
        args=args,                # 完整的配置参数对象
        root_path=args.root_path, # 数据根目录
        data_path=args.data_path, # 数据文件名
        flag=flag,                # 数据集类型：train/val/test
        size=[args.seq_len, args.label_len, args.pred_len],  # [输入长度, 标签长度, 预测长度]
        features=args.features,   # 特征类型：M/S/MS
        target=args.target,       # 目标变量列名
        timeenc=timeenc,          # 时间编码方式：0 或 1
        freq=args.freq,           # 时间频率：h/t/s/m/a/w/d/b
    )

    # ==================== 创建数据加载器 ====================
    # DataLoader 负责批次化、打乱和多进程加载
    data_loader = DataLoader(
        data_set,                    # 数据集对象
        batch_size=batch_size,       # 每个批次的样本数
        shuffle=shuffle_flag,        # 是否打乱数据顺序
        num_workers=args.num_workers, # 多进程加载的工作进程数（0 表示主进程加载）
        drop_last=drop_last,         # 是否丢弃最后不完整的批次
    )
    
    # 返回数据集对象和数据加载器
    # data_set 可用于直接访问数据和获取数据集信息
    # data_loader 用于训练时的批次迭代
    return data_set, data_loader

