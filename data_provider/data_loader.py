"""
data_loader.py - 数据集加载模块

本模块定义了用于时间序列预测的数据集类，继承自 PyTorch 的 Dataset 类。
主要功能包括：
1. 读取 CSV 格式的时间序列数据
2. 数据标准化（Z-score 归一化）
3. 时间特征提取（月、日、星期、小时等）
4. 滑动窗口采样，生成训练/验证/测试样本

数据集类型：
    - Dataset_ETT_hour: 小时级 ETT 数据集（固定划分比例）
    - Dataset_ETT_minute: 分钟级 ETT 数据集（固定划分比例）
    - Dataset_Custom: 自定义数据集（按 6:3:1 比例划分）

数据流说明：
    输入序列 (seq_x):      |<--- seq_len --->|
    标签序列 (seq_y):                |<- label_len ->|<- pred_len ->|
    
    时间轴:    ----[seq_x]----[overlap]----[pred]----
                            |<-------- seq_y ------->|
    
    其中 overlap 部分（label_len）是编码器和解码器的重叠区域，
    作为解码器的起始信息，帮助模型更好地预测。

CSV 数据格式要求：
    - 第一列必须是 'date' 列（日期时间字符串）
    - 其余列为特征列
    - 目标变量列名由 target 参数指定（默认 'OT'）
"""

# ==================== 导入依赖模块 ====================
import os                           # 文件路径操作
import numpy as np                  # 数值计算
import pandas as pd                 # 数据处理
from torch.utils.data import Dataset    # PyTorch 数据集基类
from sklearn.preprocessing import StandardScaler  # Z-score 标准化
from utils.timefeatures import time_features     # 时间特征提取工具


class Dataset_ETT_hour(Dataset):
    """
    ETT 小时级数据集类
    
    ETT（Electricity Transformer Temperature）是电力变压器温度数据集，
    包含油温和其他负载特征，常用于时间序列预测任务的基准测试。
    
    数据划分方式（固定长度）：
        - 训练集: 前 12 个月（12 * 30 * 24 = 8640 小时）
        - 验证集: 接下来 4 个月（4 * 30 * 24 = 2880 小时）
        - 测试集: 最后 4 个月（4 * 30 * 24 = 2880 小时）
    
    属性:
        seq_len (int): 输入序列长度（编码器输入）
        label_len (int): 标签长度（解码器起始部分，与编码器重叠）
        pred_len (int): 预测长度（需要预测的未来时间步数）
        data_x (ndarray): 输入特征数据
        data_y (ndarray): 目标特征数据
        data_stamp (ndarray): 时间特征数据
    """
    
    def __init__(self, args, root_path, flag="train", size=None,
                 features="S", data_path="ETTh1.csv",
                 target="OT", scale=True, timeenc=1, freq="h", seasonal_patterns=None):
        """
        初始化 ETT 小时级数据集
        
        参数:
            args: 配置参数对象
            root_path (str): 数据根目录
            flag (str): 数据集类型 'train'/'val'/'test'
            size (list): [seq_len, label_len, pred_len]，默认 [96, 48, 48]
            features (str): 特征类型
                           'M' - 多变量预测多变量
                           'S' - 单变量预测单变量（只使用目标变量）
                           'MS' - 多变量预测单变量（使用所有特征预测目标变量）
            data_path (str): 数据文件名
            target (str): 目标变量列名
            scale (bool): 是否进行标准化
            timeenc (int): 时间编码方式
                          0 - 提取月、日、星期、小时等整数特征
                          1 - 使用 time_features 提取周期性特征
            freq (str): 时间频率（'h' 小时, 't' 分钟等）
            seasonal_patterns: 季节性模式（当前未使用）
        """
        self.args = args
        
        # 设置序列长度参数
        if size is None:
            # 默认值：seq_len=96*4=384, label_len=96, pred_len=96
            self.seq_len = 24 * 4 * 4   # 16 天 = 384 小时
            self.label_len = 24 * 4     # 4 天 = 96 小时
            self.pred_len = 24 * 4      # 4 天 = 96 小时
        else:
            self.seq_len = size[0]      # 输入序列长度
            self.label_len = size[1]    # 标签长度（编码器-解码器重叠）
            self.pred_len = size[2]     # 预测长度

        # 验证数据集类型参数
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]  # 转换为数字索引

        # 保存配置参数
        self.features = features    # 特征模式
        self.target = target        # 目标变量列名
        self.scale = scale          # 是否标准化
        self.timeenc = timeenc      # 时间编码方式
        self.freq = freq            # 时间频率

        self.root_path = root_path  # 数据根目录
        self.data_path = data_path  # 数据文件名

        # 读取并预处理数据
        self.__read_data__()

    def __read_data__(self):
        """
        读取并预处理数据
        
        处理流程：
        1. 读取 CSV 文件
        2. 根据数据集类型确定数据边界
        3. 选择特征列
        4. 数据标准化（仅使用训练集统计量）
        5. 提取时间特征
        """
        # 创建标准化器
        self.scaler = StandardScaler()
        
        # 读取原始 CSV 数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ==================== 数据集边界定义 ====================
        # ETT 小时级数据集的固定划分方式
        # 12 * 30 * 24 = 8640 小时 ≈ 12 个月
        # 4 * 30 * 24 = 2880 小时 ≈ 4 个月
        
        # border1s: 各数据集的起始索引
        # 注意：验证集和测试集的起始位置需要向前移动 seq_len，
        # 以确保第一个样本有完整的输入序列
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        
        # border2s: 各数据集的结束索引
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        
        # |<-- 训练集 12 个月 -->|<-- 验证集 4 个月 -->|<-- 测试集 4 个月 -->|
        # 0                    8640                11520               14400

        # 获取当前数据集类型的边界
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ==================== 特征选择 ====================
        if self.features in ("M", "MS"):
            # M 或 MS 模式：使用所有特征列（排除第一列 date）
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            # S 模式（单变量）：只使用目标变量列
            df_data = df_raw[[self.target]]

        # ==================== 数据标准化 ====================
        if self.scale:
            # 重要：只使用训练集数据来拟合 scaler，避免数据泄露
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # 对所有数据进行标准化转换
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ==================== 时间特征提取 ====================
        # 获取当前数据集范围内的日期列
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            # 方式 0：提取基本的时间整数特征
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            # 方式 1：使用 time_features 提取周期性编码特征
            # 返回形状为 [num_features, seq_len]，需要转置
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)  # 转为 [seq_len, num_features]

        # 保存处理后的数据
        self.data_x = data[border1:border2]   # 输入特征
        self.data_y = data[border1:border2]   # 目标特征（与 data_x 相同，在采样时区分）
        self.data_stamp = data_stamp          # 时间特征

    def __getitem__(self, index):
        """
        获取单个样本（滑动窗口采样）
        
        采样逻辑示意图：
            输入序列 seq_x:     |<-------- seq_len -------->|
            目标序列 seq_y:              |<- label_len ->|<- pred_len ->|
            
            时间轴索引:   s_begin...............s_end
                                      r_begin...............r_end
        
        参数:
            index (int): 样本索引（滑动窗口的起始位置）
        
        返回:
            seq_x: 编码器输入序列 [seq_len, num_features]
            seq_y: 解码器目标序列 [label_len + pred_len, num_features]
            seq_x_mark: 编码器时间特征 [seq_len, time_features]
            seq_y_mark: 解码器时间特征 [label_len + pred_len, time_features]
        """
        # 计算输入序列的起止索引
        s_begin = index
        s_end = s_begin + self.seq_len
        
        # 计算目标序列的起止索引
        # r_begin 从 s_end 向前移动 label_len，实现重叠
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # 提取输入序列和目标序列
        seq_x = self.data_x[s_begin:s_end]       # 编码器输入
        seq_y = self.data_y[r_begin:r_end]       # 解码器目标（包含 label + pred）
        seq_x_mark = self.data_stamp[s_begin:s_end]      # 编码器时间特征
        seq_y_mark = self.data_stamp[r_begin:r_end]      # 解码器时间特征

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        返回数据集中的样本数量
        
        滑动窗口的总数 = 数据长度 - seq_len - pred_len + 1
        确保每个样本都有完整的输入序列和预测目标
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        反标准化：将标准化后的数据转换回原始尺度
        
        参数:
            data (ndarray): 标准化后的数据
        
        返回:
            ndarray: 原始尺度的数据
        """
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    """
    ETT 分钟级数据集类
    
    与 Dataset_ETT_hour 类似，但数据采样频率为 15 分钟（每小时 4 个点）。
    
    数据划分方式（固定长度）：
        - 训练集: 前 12 个月（12 * 30 * 24 * 4 = 34560 个点）
        - 验证集: 接下来 4 个月（4 * 30 * 24 * 4 = 11520 个点）
        - 测试集: 最后 4 个月（4 * 30 * 24 * 4 = 11520 个点）
    
    与小时级数据集的区别：
        1. 数据点密度是小时级的 4 倍
        2. 时间特征额外包含 minute 分钟字段
    """
    
    def __init__(self, args, root_path, flag="train", size=None,
                 features="S", data_path="ETTm1.csv",
                 target="OT", scale=True, timeenc=1, freq="t", seasonal_patterns=None):
        """
        初始化 ETT 分钟级数据集
        
        参数与 Dataset_ETT_hour 相同，主要区别：
            - freq 默认为 't'（分钟）
            - data_path 默认为 'ETTm1.csv'
        """
        self.args = args
        
        # 设置序列长度参数
        if size is None:
            self.seq_len = 24 * 4 * 4   # 默认值
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # 验证数据集类型
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        # 保存配置参数
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # 读取并预处理数据
        self.__read_data__()

    def __read_data__(self):
        """
        读取并预处理分钟级数据
        
        与小时级数据的主要区别：
        1. 数据边界计算时乘以 4（每小时 4 个数据点）
        2. 时间特征额外提取 minute 字段
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ==================== 数据集边界定义 ====================
        # 分钟级数据，每小时 4 个点（15 分钟间隔）
        # 12 * 30 * 24 * 4 = 34560 个点 ≈ 12 个月
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ==================== 特征选择 ====================
        if self.features in ("M", "MS"):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            df_data = df_raw[[self.target]]

        # ==================== 数据标准化 ====================
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ==================== 时间特征提取 ====================
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            # 分钟级数据额外提取 minute 字段
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)  # 新增分钟特征
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 保存处理后的数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        获取单个样本（滑动窗口采样）
        
        采样逻辑与 Dataset_ETT_hour 完全相同
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """反标准化"""
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    """
    自定义数据集类
    
    用于加载用户自己的 CSV 格式时间序列数据。
    
    与 ETT 数据集的主要区别：
        1. 数据按比例划分（60% 训练，30% 验证，10% 测试）
        2. 自动将目标变量移到最后一列
        3. 不依赖固定的数据长度
    
    CSV 格式要求：
        - 必须包含 'date' 列（日期时间）
        - 必须包含 target 指定的目标变量列
        - 其他列作为特征变量
    
    使用示例：
        # CSV 文件格式：
        # date, feature1, feature2, OT
        # 2020-01-01 00:00:00, 1.2, 3.4, 5.6
        # ...
    """
    
    def __init__(self, args, root_path, flag="train", size=None,
                 features="S", data_path="custom.csv",
                 target="OT", scale=True, timeenc=1, freq="h", seasonal_patterns=None):
        """
        初始化自定义数据集
        
        参数与 Dataset_ETT_hour 相同，主要区别：
            - data_path: 用户自定义的 CSV 文件名
            - 数据划分比例：训练 60%，验证 30%，测试 10%
        """
        self.args = args
        
        # 设置序列长度参数
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # 验证数据集类型
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        # 保存配置参数
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # 读取并预处理数据
        self.__read_data__()

    def __read_data__(self):
        """
        读取并预处理自定义数据
        
        与 ETT 数据集的区别：
        1. 按比例（而非固定长度）划分数据集，比例为 6:3:1
        2. 重新排列列顺序，将目标变量放到最后
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ==================== 重新排列列顺序 ====================
        # 将目标变量移到最后一列，方便后续处理
        # 最终列顺序: [date, feature1, feature2, ..., target]
        cols = list(df_raw.columns)
        cols.remove(self.target)  # 移除目标变量
        cols.remove("date")       # 移除日期列
        df_raw = df_raw[["date"] + cols + [self.target]]  # 重新排列

        # ==================== 按比例划分数据集 ====================
        # 训练集 : 验证集 : 测试集 = 6 : 3 : 1
        total_len = len(df_raw)
        num_train = int(total_len * 0.6)
        num_vali = int(total_len * 0.3)
        num_test = total_len - num_train - num_vali  # 剩余部分给测试集，保证总长度一致

        # 计算各数据集的边界索引
        # 验证集和测试集的起始位置需要向前移动 seq_len
        border1s = [0, num_train - self.seq_len, num_train + num_vali - self.seq_len]
        border2s = [num_train, num_train + num_vali, total_len]
        
        # |<-- 训练集 60% -->|<-- 验证集 30% -->|<-- 测试集 10% -->|

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ==================== 特征选择 ====================
        if self.features in ("M", "MS"):
            # 使用所有特征列（排除 date 列）
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        else:
            # 只使用目标变量
            df_data = df_raw[[self.target]]

        # ==================== 数据标准化 ====================
        if self.scale:
            # 仅使用训练集数据拟合 scaler
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ==================== 时间特征提取 ====================
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            # 提取基本时间特征（月、日、星期、小时）
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        else:
            # 使用周期性编码
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 保存处理后的数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        获取单个样本（滑动窗口采样）
        
        采样逻辑与 Dataset_ETT_hour 完全相同
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """返回数据集样本数量"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """反标准化"""
        return self.scaler.inverse_transform(data)
