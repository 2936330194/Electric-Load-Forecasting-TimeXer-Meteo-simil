import h5py
import sys
import os

def analyze_h5(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    print("=" * 60)
    print(f"📄 分析文件: {os.path.basename(file_path)}")
    print(f"📦 文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print(f"📂 绝对路径: {os.path.abspath(file_path)}")
    print("=" * 60)

    try:
        with h5py.File(file_path, 'r') as f:
            # ==========================================
            # 1. 打印全局属性 (Global Attributes)
            # ==========================================
            print("\n[全局配置属性 (Global Attributes)]")
            if not f.attrs.keys():
                print("  (无全局属性)")
            else:
                for key in f.attrs.keys():
                    print(f"  - {key}: {f.attrs[key]}")

            # ==========================================
            # 2. 遍历并打印内部所有数据集结构
            # ==========================================
            print("\n[数据集结构 (Datasets)]")
            
            def print_dataset_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"\n  ▶ /{name}")
                    print(f"      维数/形状 (Shape): {obj.shape}")
                    print(f"      数据类型 (Dtype): {obj.dtype}")
                    
                    # 打印单数据集自带的局部属性
                    for key in obj.attrs.keys():
                        print(f"      [属性] {key}: {obj.attrs[key]}")
                    
                    # ---------- 专门适配项目文件结构的数据解析 ----------
                    if name in ["timestamps", "time"]:
                        try:
                            # 提取头尾时间戳以计算整体跨度
                            ts_data = obj[:]
                            if len(ts_data) > 0:
                                # 处理 bytes 解码
                                first_ts = ts_data[0].decode('utf-8') if isinstance(ts_data[0], bytes) else str(ts_data[0])
                                last_ts  = ts_data[-1].decode('utf-8') if isinstance(ts_data[-1], bytes) else str(ts_data[-1])
                                print(f"      🕒 时间范围: {first_ts}  ~  {last_ts}")
                                print(f"      🕒 时间步总数: {len(ts_data)} 步")
                        except Exception as e:
                            print(f"      🕒 时间范围解析失败: {e}")
                    
                    if name == "variables" or name == "channels":
                        try:
                            # 提取通道信息
                            vars_data = obj[:]
                            var_names = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in vars_data]
                            print(f"      📊 通道数: {len(var_names)}")
                            print(f"      📊 通道名称: [ {', '.join(var_names)} ]")
                        except Exception as e:
                            print(f"      📊 通道解析失败: {e}")
                            
            # 递归遍历所有组与数据集
            f.visititems(print_dataset_info)
            print("\n" + "=" * 60)
            
    except Exception as e:
        print(f"\n❌ 解析 HDF5 文件时发生错误: {e}")

if __name__ == "__main__":
    # 在这里直接指定您想要分析的 HDF5 文件路径
    # 提示：Windows 路径建议在字符串前加上 r，或者把 \ 改为 /
    # 例如： TARGET_H5_FILE = r"D:\您的路径\hunan_grid_2025-01-01_to_2025-01-10.h5"
    TARGET_H5_FILE = r"D:\Pycharm Project\Scientific Reasearch\Electricity Forecasting\Electric Load Forecasting TimeXer Meteo simil\data\hunan_grid_2024_2025_filtered_15min.h5"
    
    if TARGET_H5_FILE.strip() == "":
        print("💡 提示：请先在脚本末尾的 TARGET_H5_FILE 变量中填入您想要分析的 .h5 文件的具体路径，然后再点击运行！")
    else:
        analyze_h5(TARGET_H5_FILE)
