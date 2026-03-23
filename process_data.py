import pandas as pd
import os

def main():
    input_file = '湖南省电力负荷（25.10.25-26.2.26).xlsx'
    output_file = '湖南省电力负荷_processed（25.10.25-26.2.26).csv'

    print(f"正在读取 {input_file}...")
    df = pd.read_excel(input_file)

    # 1. 将“时间”列中的中文冒号替换为英文冒号
    df['时间'] = df['时间'].astype(str).str.replace('：', ':')

    # 2. 将“日期”列标准化提取出纯日期部分
    # 通过 to_datetime 并用 dt.strftime 提取 '%Y-%m-%d' 防止 Excel 本身存在的 datetime 格式问题
    # 2. 将“日期”列标准化提取出纯日期部分
    def robust_date_parse(d):
        if hasattr(d, 'strftime'):
            return d.strftime('%Y-%m-%d')
        # 如果是字符串，尝试用 pandas 解析
        try:
            return pd.to_datetime(str(d)).strftime('%Y-%m-%d')
        except:
            return pd.to_datetime(str(d), format='mixed', dayfirst=True).strftime('%Y-%m-%d')
            
    date_str = df['日期'].apply(robust_date_parse)
    
    # 3. 处理时间中的 24:00 问题
    # Pandas 无法直接解析 24:00，这通常表示第二天的 00:00
    is_24_00 = df['时间'] == '24:00'
    time_str = df['时间'].replace('24:00', '00:00')

    # 4. 合并日期和时间，重新转为 datetime 对象以便使用
    combined_datetime = pd.to_datetime(date_str + ' ' + time_str, format='mixed')
    
    # 将原来是 24:00 的行加上一天
    if is_24_00.any():
        combined_datetime[is_24_00] += pd.Timedelta(days=1)

    # 5. 将时间格式化为用户要求的 yyyy/m/d h:mm 
    # 为了保证不同操作系统下的兼容性（避免 %-m 或 %#m 的差异），这里直接使用格式化字符串
    df['date'] = combined_datetime.apply(lambda dt: f"{dt.year}/{dt.month}/{dt.day} {dt.hour}:{dt.minute:02d}")

    # 5. 找出代表负荷的列并将其重命名为 load
    # 原数据中一般可能为 '实际负荷' 等
    if '实际负荷' in df.columns:
        df.rename(columns={'实际负荷': 'load'}, inplace=True)
    else:
        # 如果列名不是"实际负荷"，则寻找其他非日期/时间的列
        for col in df.columns:
            if col not in ['日期', '时间', 'date']:
                df.rename(columns={col: 'load'}, inplace=True)
                break

    # 6. 删除“日期”和“时间”列，只保留 date 和 load 两列
    final_df = df[['date', 'load']]

    print(f"正在保存到 {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("处理完成！")

if __name__ == "__main__":
    main()
