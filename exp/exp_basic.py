"""
exp_basic.py - 实验基类模块

本模块定义了所有实验类的基类 Exp_Basic，提供了：
1. 自动扫描 models 目录下的模型文件
2. 惰性加载模型类（按需导入，节省内存）
3. 设备管理（GPU/CPU 选择）
4. 模型构建的统一接口

设计模式：
- 模板方法模式：_build_model() 由子类实现具体逻辑
- 惰性加载模式：LazyModelDict 实现按需导入模型

使用方式：
    class Exp_Main(Exp_Basic):
        def _build_model(self):
            model = self.model_dict[self.args.model].Model(self.args)
            return model
"""

import os          # 用于文件和目录操作
import importlib   # 用于动态导入模块


class Exp_Basic:
    """
    实验基类
    
    所有具体实验类（如 Exp_Main）都应继承此类。
    提供模型加载、设备选择等基础功能。
    
    属性:
        args: 命令行参数对象，包含模型配置、训练参数等
        model_dict: LazyModelDict 实例，用于惰性加载模型类
        device: 计算设备（cuda 或 cpu）
        model: 构建好的模型实例
    """
    
    def __init__(self, args):
        """
        初始化实验基类
        
        参数:
            args: 命令行参数对象，通常由 argparse 解析得到
                  必须包含以下属性：
                  - use_gpu (bool): 是否使用 GPU
                  - gpu (int): GPU 设备编号
                  - device (str): 设备字符串，如 'cuda:0' 或 'cpu'
        """
        self.args = args  # 保存参数配置
        
        # 扫描 models 目录，构建模型名称到模块路径的映射
        # 然后创建惰性加载字典
        self.model_dict = LazyModelDict(self._scan_models_directory())
        
        # 根据配置获取计算设备（GPU 或 CPU）
        self.device = self._acquire_device()
        
        # 构建模型并将其移动到指定设备上
        # _build_model() 是抽象方法，由子类实现
        self.model = self._build_model().to(self.device)

    def _scan_models_directory(self):
        """
        扫描 models 目录，获取所有可用模型的映射
        
        遍历 models 目录下的所有 .py 文件（排除 __init__.py），
        构建模型名称（文件名去掉 .py 后缀）到完整模块路径的映射。
        
        返回:
            dict: 模型名称到模块路径的映射
                  例如: {'DLinear': 'models.DLinear', 
                        'Autoformer': 'models.Autoformer'}
        """
        model_map = {}
        models_dir = "models"  # 模型目录名称
        
        # 检查 models 目录是否存在
        if os.path.exists(models_dir):
            # 遍历目录下的所有文件
            for filename in os.listdir(models_dir):
                # 只处理 .py 文件，排除 __init__.py
                if filename.endswith(".py") and filename != "__init__.py":
                    # 去掉 .py 后缀得到模块名
                    # 例如: 'DLinear.py' -> 'DLinear'
                    module_name = filename[:-3]
                    
                    # 构建完整的模块导入路径
                    # 例如: 'models.DLinear'
                    full_path = f"{models_dir}.{module_name}"
                    
                    # 添加到映射字典
                    model_map[module_name] = full_path
                    
        return model_map

    def _build_model(self):
        """
        构建模型的抽象方法（模板方法模式）
        
        子类必须重写此方法以实现具体的模型构建逻辑。
        
        抛出:
            NotImplementedError: 如果子类未实现此方法
            
        返回:
            model: 构建好的模型实例
        """
        raise NotImplementedError

    def _acquire_device(self):
        """
        获取计算设备
        
        根据 args.use_gpu 参数决定使用 GPU 还是 CPU。
        如果使用 GPU，会设置 CUDA_VISIBLE_DEVICES 环境变量。
        
        返回:
            device: 设备字符串，如 'cuda:0' 或 'cpu'
        """
        if self.args.use_gpu:
            # 设置可见的 CUDA 设备，限制程序只能看到指定的 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = self.args.device
            print(f"Use GPU: {device}")
        else:
            device = self.args.device
            print("Use CPU")
        return device


class LazyModelDict(dict):
    """
    惰性加载模型字典
    
    继承自 dict，实现按需导入模型类的功能。
    只有在首次访问某个模型时才会导入对应的模块，
    避免一次性导入所有模型造成的内存浪费和启动延迟。
    
    设计思想:
        - 惰性加载（Lazy Loading）：延迟导入，按需加载
        - 缓存机制：导入后的模型类会被缓存，避免重复导入
    
    属性:
        model_map (dict): 模型名称到模块路径的映射
    
    使用示例:
        model_dict = LazyModelDict({'DLinear': 'models.DLinear'})
        model_class = model_dict['DLinear']  # 此时才导入 DLinear 模块
        model = model_class(args)  # 创建模型实例
    """
    
    def __init__(self, model_map):
        """
        初始化惰性模型字典
        
        参数:
            model_map (dict): 模型名称到模块路径的映射
                             例如: {'DLinear': 'models.DLinear'}
        """
        self.model_map = model_map  # 保存模型映射
        super().__init__()          # 调用父类 dict 的初始化方法

    def __getitem__(self, key):
        """
        重写字典的取值方法，实现惰性加载
        
        当访问 model_dict[key] 时：
        1. 如果该模型类已经加载过（在缓存中），直接返回
        2. 如果未加载过，则动态导入对应模块并获取模型类
        3. 将模型类缓存到字典中，供后续使用
        
        参数:
            key (str): 模型名称，如 'DLinear'
            
        返回:
            model_class: 模型类（不是实例）
            
        抛出:
            NotImplementedError: 如果找不到对应的模型
            AttributeError: 如果模块中没有 'Model' 类或同名类
        """
        # 检查是否已经加载并缓存了该模型类
        if key in self:
            return super().__getitem__(key)
        
        # 检查请求的模型是否在扫描到的模型列表中
        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")

        # 获取模块的完整导入路径
        # 例如: 'models.DLinear'
        module_path = self.model_map[key]
        
        # 动态导入模块
        # 等价于: import models.DLinear as module
        module = importlib.import_module(module_path)
        
        # 尝试获取模型类
        # 优先查找名为 'Model' 的类（标准命名）
        if hasattr(module, "Model"):
            model_class = module.Model
        # 其次查找与模块同名的类
        # 例如: 在 DLinear.py 中查找 DLinear 类
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")
        
        # 将模型类缓存到字典中，下次访问时直接返回
        self[key] = model_class
        
        return model_class
