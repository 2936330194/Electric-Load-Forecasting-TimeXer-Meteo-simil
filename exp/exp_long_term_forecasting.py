"""
exp_long_term_forecasting.py - 长期时间序列预测实验模块

本模块定义了长期预测实验类 Exp_Long_Term_Forecast，用于：
1. 时间序列预测模型的训练、验证和测试
2. 支持多种预测模型（DLinear、Autoformer、Informer 等）
3. 实现早停机制和学习率调整策略
4. 支持混合精度训练（AMP）和多 GPU 并行训练

核心流程：
    1. train(): 训练模型，使用早停机制防止过拟合
    2. vali(): 在验证集上评估模型性能
    3. test(): 在测试集上进行最终评估并保存结果

数据流说明：
    - batch_x: 编码器输入序列 [batch, seq_len, features]
    - batch_y: 包含 label_len + pred_len 的目标序列
    - batch_x_mark: 编码器时间特征标记
    - batch_y_mark: 解码器时间特征标记
    - dec_inp: 解码器输入，由 label_len 部分真实值 + pred_len 部分零填充组成
"""

# ==================== 导入依赖模块 ====================
from data_provider.data_factory import data_provider  # 数据提供器，用于获取数据集和数据加载器
from exp.exp_basic import Exp_Basic                   # 实验基类，提供设备管理和模型加载功能
from utils.tools import EarlyStopping, adjust_learning_rate  # 早停机制和学习率调整工具
from utils.metrics import cal_eval                    # 评估指标计算函数（MAE, MSE, RMSE 等）

import torch                  # PyTorch 深度学习框架
import torch.nn as nn         # 神经网络模块
from torch import optim       # 优化器模块
import os                     # 操作系统接口
import time                   # 时间模块，用于计时
import numpy as np            # 数值计算库


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    长期时间序列预测实验类
    
    继承自 Exp_Basic，实现完整的训练、验证和测试流程。
    适用于长期预测任务（如电力负荷预测、天气预测等）。
    
    主要特性:
        - 支持多种时序预测模型
        - 早停机制防止过拟合
        - 混合精度训练加速
        - 多 GPU 并行训练
        - 自动保存最佳模型
    
    典型使用流程:
        exp = Exp_Long_Term_Forecast(args)
        exp.train(setting)      # 训练模型
        exp.test(setting)       # 测试模型
    """
    
    def __init__(self, args):
        """
        初始化长期预测实验
        
        参数:
            args: 包含所有实验配置的参数对象，主要包括：
                  - model (str): 模型名称，如 'DLinear'
                  - seq_len (int): 输入序列长度
                  - label_len (int): 解码器起始标签长度
                  - pred_len (int): 预测长度
                  - features (str): 特征类型，'M'/'S'/'MS'
                  - learning_rate (float): 学习率
                  - train_epochs (int): 训练轮数
                  - patience (int): 早停耐心值
                  - use_amp (bool): 是否使用混合精度训练
                  - use_multi_gpu (bool): 是否使用多 GPU
        """
        # 调用父类构造函数，完成模型构建和设备初始化
        super().__init__(args)

    def _build_model(self):
        """
        构建预测模型
        
        从 model_dict 中获取指定的模型类并实例化。
        如果配置了多 GPU，则使用 DataParallel 进行包装。
        
        返回:
            model: 构建好的模型实例
        """
        # 从惰性加载字典中获取模型类，传入参数并实例化
        # .float() 确保模型使用 32 位浮点数
        model = self.model_dict[self.args.model](self.args).float()
        
        # 多 GPU 并行训练配置
        if self.args.use_multi_gpu and self.args.use_gpu:
            # DataParallel 自动将数据分发到多个 GPU
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        获取数据集和数据加载器
        
        参数:
            flag (str): 数据集类型标志
                       'train' - 训练集
                       'val'   - 验证集  
                       'test'  - 测试集
        
        返回:
            data_set: 数据集对象
            data_loader: 数据加载器
        """
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        """
        选择优化器
        
        当前使用 Adam 优化器，具有自适应学习率特性。
        
        返回:
            optimizer: Adam 优化器实例
        """
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        """
        选择损失函数
        
        当前使用均方误差损失（MSE Loss），适用于回归任务。
        
        返回:
            criterion: MSE 损失函数
        """
        return nn.MSELoss()

    def _inverse_transform_for_eval(self, preds, trues, data_set):
        scaler = getattr(data_set, "scaler", None)
        if scaler is None or not hasattr(scaler, "mean_"):
            return None, None

        if self.args.features == "MS":
            mean = scaler.mean_[-1]
            scale = scaler.scale_[-1]
            preds_inv = preds * scale + mean
            trues_inv = trues * scale + mean
            return preds_inv, trues_inv

        shape = preds.shape
        preds_2d = preds.reshape(-1, shape[-1])
        trues_2d = trues.reshape(-1, shape[-1])
        preds_inv = scaler.inverse_transform(preds_2d).reshape(shape)
        trues_inv = scaler.inverse_transform(trues_2d).reshape(shape)
        return preds_inv, trues_inv

    def _model_forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """
        统一模型前向接口。
        """
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def vali(self, vali_data, vali_loader, criterion):
        """
        验证模型性能
        
        在验证集或测试集上评估模型，计算平均损失。
        此方法不更新模型参数。
        
        参数:
            vali_data: 验证数据集对象
            vali_loader: 验证数据加载器
            criterion: 损失函数
        
        返回:
            float: 验证集上的平均损失值
        """
        total_loss = []  # 存储每个 batch 的损失
        
        # 设置模型为评估模式（关闭 Dropout、BatchNorm 使用移动平均）
        self.model.eval()
        
        # 禁用梯度计算，节省内存和计算
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                # ==================== 数据转移到设备 ====================
                batch_x = batch_x.float().to(self.device)       # 编码器输入 [B, seq_len, D]
                batch_y = batch_y.float().to(self.device)       # 目标序列 [B, label_len+pred_len, D]
                batch_x_mark = batch_x_mark.float().to(self.device)  # 编码器时间特征
                batch_y_mark = batch_y_mark.float().to(self.device)  # 解码器时间特征
                # ==================== 构建解码器输入 ====================
                # 解码器输入由两部分组成：
                # 1. label_len 部分的真实值（作为起始信息）
                # 2. pred_len 部分的零填充（待预测部分）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ==================== 模型前向传播 ====================
                if self.args.use_amp:
                    # 混合精度推理
                    with torch.cuda.amp.autocast():
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # ==================== 提取预测部分 ====================
                # f_dim: 特征维度起始索引
                # 'MS' (多变量预测单变量): 只取最后一个特征（通常是目标变量）
                # 其他: 取所有特征
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 取预测长度部分
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]  # 对应的真实值

                # 计算损失
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        # 恢复模型为训练模式
        self.model.train()
        
        # 返回平均损失
        return np.average(total_loss)

    def train(self, setting):
        """
        训练模型
        
        完整的训练流程，包括：
        1. 加载训练、验证、测试数据
        2. 初始化优化器和早停机制
        3. 多轮训练，每轮结束后验证
        4. 早停检查和学习率调整
        5. 加载最佳模型
        
        参数:
            setting (str): 实验设置字符串，用于区分不同实验配置
                          例如: 'DLinear_ETTh1_96_192_pl96_dm512'
        
        返回:
            model: 训练完成后加载的最佳模型
        """
        # ==================== 加载数据 ====================
        train_data, train_loader = self._get_data(flag="train")  # 训练集
        vali_data, vali_loader = self._get_data(flag="val")      # 验证集
        test_data, test_loader = self._get_data(flag="test")     # 测试集

        # ==================== 创建模型保存目录 ====================
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # ==================== 初始化训练组件 ====================
        time_now = time.time()                  # 记录训练开始时间
        train_steps = len(train_loader)         # 每个 epoch 的训练步数
        
        # 早停机制：当验证损失连续 patience 个 epoch 不下降时停止训练
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()  # 优化器
        criterion = self._select_criterion()    # 损失函数

        # 混合精度训练：使用 GradScaler 进行梯度缩放
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # ==================== 开始训练循环 ====================
        for epoch in range(self.args.train_epochs):
            iter_count = 0      # 迭代计数器（用于计算速度）
            train_loss = []     # 存储当前 epoch 的所有损失
            
            self.model.train()  # 设置模型为训练模式
            epoch_time = time.time()  # 记录当前 epoch 开始时间

            # 遍历训练数据
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                # 清空梯度（PyTorch 梯度默认累积）
                model_optim.zero_grad()

                # ==================== 数据转移到设备 ====================
                batch_x = batch_x.float().to(self.device)       # 编码器输入
                batch_y = batch_y.float().to(self.device)       # 目标序列
                batch_x_mark = batch_x_mark.float().to(self.device)  # 编码器时间特征
                batch_y_mark = batch_y_mark.float().to(self.device)  # 解码器时间特征
                # ==================== 构建解码器输入 ====================
                # 结构: [label_len 部分真实值] + [pred_len 部分零填充]
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ==================== 前向传播与反向传播 ====================
                if self.args.use_amp:
                    # 混合精度训练路径
                    with torch.cuda.amp.autocast():
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                    
                    # 使用 scaler 进行梯度缩放和更新
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 标准训练路径
                    outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    
                    # 标准反向传播
                    loss.backward()
                    model_optim.step()

                # 记录损失
                train_loss.append(loss.item())

                # ==================== 打印训练进度（每 100 步） ====================
                if (i + 1) % 100 == 0:
                    # 计算训练速度
                    speed = (time.time() - time_now) / iter_count
                    # 估算剩余时间
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    
                    # 重置计数器
                    iter_count = 0
                    time_now = time.time()

            # ==================== Epoch 结束处理 ====================
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.4f}s")
            
            # 计算平均训练损失
            train_loss = np.average(train_loss)
            
            # 在验证集和测试集上评估
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                f"Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}"
            )

            # ==================== 早停检查 ====================
            # 如果验证损失改善，保存模型；否则增加计数器
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 调整学习率（如果配置了学习率衰减策略）
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # ==================== 加载最佳模型 ====================
        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0):
        """
        测试模型并保存结果
        
        在测试集上运行模型，计算评估指标并保存预测结果。
        
        参数:
            setting (str): 实验设置字符串，用于创建结果保存目录
            test (int): 测试标志
                       0 - 训练后直接测试（使用当前内存中的模型）
                       1 - 仅测试模式（从 checkpoint 加载模型）
        
        保存文件:
            - results/{setting}/pred.npy: 模型预测值
            - results/{setting}/true.npy: 真实值
        
        打印指标:
            - MSE (均方误差)
            - MAE (平均绝对误差)
            - RMSE (均方根误差)
            - MAPE (平均绝对百分比误差)
        """
        # ==================== 加载模型权重（仅测试模式） ====================
        if test:
            # 从 checkpoint 目录加载已保存的最佳模型
            checkpoint_path = os.path.join(self.args.checkpoints, setting, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                print(f"Loading model from: {checkpoint_path}")
                checkpoint_state = torch.load(checkpoint_path, map_location=self.device)
                try:
                    self.model.load_state_dict(checkpoint_state)
                except RuntimeError as e:
                    print(f"Strict checkpoint loading failed: {e}")
                    print("Fallback to strict=False loading for compatibility.")
                    load_result = self.model.load_state_dict(checkpoint_state, strict=False)
                    print(f"Missing keys: {load_result.missing_keys}")
                    print(f"Unexpected keys: {load_result.unexpected_keys}")
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                print("Please ensure the model has been trained and saved.")
                return
        
        # 获取测试数据
        test_data, test_loader = self._get_data(flag="test")

        # 存储所有预测值和真实值
        preds = []
        trues = []

        # 设置模型为评估模式
        self.model.eval()
        
        # 禁用梯度计算

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                # ==================== 数据转移到设备 ====================
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # ==================== 构建解码器输入 ====================
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # ==================== 模型预测 ====================
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # ==================== 提取预测部分 ====================
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # 转换为 numpy 数组并存储
                pred = outputs.detach().cpu().numpy()  # 分离计算图，移到 CPU
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
        
        # 检查是否有测试样本
        if len(preds) == 0:
            print("No test samples found.")
            return

        # ==================== 合并所有 batch 的结果 ====================
        preds = np.concatenate(preds, axis=0)  # [N, pred_len, features]
        trues = np.concatenate(trues, axis=0)  # [N, pred_len, features]

        # ==================== 计算并打印评估指标 ====================
        scaled_eval = cal_eval(trues, preds)
        print("[Scaled] metrics:")
        print(scaled_eval)
        preds_inv = None
        trues_inv = None

        if getattr(self.args, "inverse_eval", False):
            preds_inv, trues_inv = self._inverse_transform_for_eval(preds, trues, test_data)
            if preds_inv is None:
                print("[Inverse] skipped (scaler not found)")
            else:
                inv_eval = cal_eval(trues_inv, preds_inv)
                print("[Inverse] metrics:")
                print(inv_eval)

        # ==================== 保存预测结果 ====================
        results_path = os.path.join("./results/", setting)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            
        # 保存预测值和真实值为 numpy 文件
        np.save(os.path.join(results_path, "pred.npy"), preds)
        np.save(os.path.join(results_path, "true.npy"), trues)

        # 保存反归一化后的预测值和真实值为 numpy 文件
        if preds_inv is not None and trues_inv is not None:
            np.save(os.path.join(results_path, "pred_inv.npy"), preds_inv)
            np.save(os.path.join(results_path, "true_inv.npy"), trues_inv)


