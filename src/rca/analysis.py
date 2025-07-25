import os
import pandas as pd
from collections import defaultdict

def RCAeval_stats(base_dir):
    # 初始化三个系统的统计字典
    ob_stats = defaultdict(lambda: defaultdict(int))
    tt_stats = defaultdict(lambda: defaultdict(int))
    ss_stats = defaultdict(lambda: defaultdict(int))
    
    # 定义数据集和系统的名称
    datasets = ['RE1', 'RE2', 'RE3']
    systems = ['OB', 'TT', 'SS']
    
    # 遍历每个数据集和系统的组合
    for dataset in datasets:
        for system in systems:
            # 构建当前文件夹路径
            folder_path = os.path.join(base_dir, f"{dataset}-{system}")
            
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                print(f"警告: 文件夹 {folder_path} 不存在")
                continue
            
            # 获取所有故障类型文件夹
            fault_folders = [f for f in os.listdir(folder_path) 
                          if os.path.isdir(os.path.join(folder_path, f))]
            
            # 遍历每个故障类型文件夹
            for folder in fault_folders:
                parts = folder.split('_')
                fault_type = parts[1] if len(parts) > 1 else folder
                fault_path = os.path.join(folder_path, folder)
                
                # 获取该故障类型下的所有案例
                cases = [f for f in os.listdir(fault_path) 
                        if os.path.isdir(os.path.join(fault_path, f)) and f in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]]
                # 更新对应系统的统计字典
                if system == 'OB':
                    ob_stats[dataset][fault_type] += len(cases)
                elif system == 'TT':
                    tt_stats[dataset][fault_type] += len(cases)
                elif system == 'SS':
                    ss_stats[dataset][fault_type] += len(cases)
    
    # 将统计结果转换为 DataFrame
    ob_df = pd.DataFrame(ob_stats).fillna(0).astype(int).T
    tt_df = pd.DataFrame(tt_stats).fillna(0).astype(int).T
    ss_df = pd.DataFrame(ss_stats).fillna(0).astype(int).T
    
    return ob_df, tt_df, ss_df

def aiops2021_stats(base_dir):
    df = pd.read_csv(base_dir)
    type_counts = df['anomaly_type'].value_counts().reset_index()
    type_counts.columns = ['故障类型', '数量']
    return type_counts

def GAIA_stats(base_dir):
    df = pd.read_csv(base_dir)
    type_counts = df['anomaly_type'].value_counts().reset_index()
    type_counts.columns = ['故障类型', '数量']
    return type_counts

if __name__ == "__main__":
    # gt_dir="data/gaia_groundtruth.csv"
    # type_counts = GAIA_stats(gt_dir)
    # print(type_counts)
    # a=""
    # for i in range(len(type_counts)):
    #     a += f"{type_counts['故障类型'][i][1:-1]}: {type_counts['数量'][i]}; "
    # print(a)
    # base_directory = "/mnt/jfs/RCAEval"
    
    # # 统计案例数量
    # ob_df, tt_df, ss_df = RCAeval_stats(base_directory)

    # # 打印统计结果
    # print("OB 系统统计结果:")
    # print(ob_df)
    # print("\nTT 系统统计结果:")
    # print(tt_df)
    # print("\nSS 系统统计结果:")
    # print(ss_df)
    dir="/mnt/jfs/RCAEval/RE1-SS/carts_cpu/1/data.csv"
    df=pd.read_csv(dir)
    print(df.head())
    print(df.columns)