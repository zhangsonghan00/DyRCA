import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob

# 需要比较的服务列表 - 可以根据实际数据调整
COMPARE_SERVICES = [
    'ts-basic-service', 
    'ts-route-plan-service',
    'ts-ui-dashboard',
    'ts-travel-plan-service',
    'ts-travel2-service',
    'ts-preserve-service',
    'ts-payment-service',
    'ts-order-service'
]

# 使用指定的五个指标类型
METRIC_TYPES = [
    'container.cpu.usage',
    'container.memory.usage',
    'k8s.pod.cpu_limit_utilization',
    'hubble_http_request_duration_p95_seconds',
    'queueSize'
]


def parse_service_and_fault_type(datapack_name):
    """从datapack名称解析服务名和故障类型"""
    # 匹配格式: ts0-ts-basic-service-response-replace-code-5djll8
    # 提取服务名(ts到service或dashboard)和故障类型
    match = re.match(r'ts\d+-((ts-.+?-(?:service|dashboard)))-(.*?)-[a-zA-Z0-9]+$', datapack_name)
    if match:
        service_name = match.group(1)
        fault_type = match.group(3)  # 修正为group(3)以正确获取故障类型
        return service_name, fault_type
    return datapack_name, "unknown"


def load_and_combine_data(datapack_folder):
    """加载并合并正常和异常数据"""
    try:
        # 加载正常数据
        normal_path = os.path.join(datapack_folder, 'normal_metrics.parquet')
        normal_df = pd.read_parquet(normal_path) if os.path.exists(normal_path) else pd.DataFrame()
        
        # 加载异常数据
        abnormal_path = os.path.join(datapack_folder, 'abnormal_metrics.parquet')
        abnormal_df = pd.read_parquet(abnormal_path) if os.path.exists(abnormal_path) else pd.DataFrame()
        
        # 标记数据类型并合并
        normal_df['data_type'] = 'normal'
        abnormal_df['data_type'] = 'abnormal'
        
        # 合并数据，正常数据在前，异常数据在后
        combined_df = pd.concat([normal_df, abnormal_df], ignore_index=True)
        combined_df['value'] = combined_df['value'].fillna(0)
        combined_df.dropna(subset=["metric", "service_name", "value"], inplace=True)
        
        # 记录故障注入点位置（正常数据的长度）
        injection_point = len(normal_df) if not normal_df.empty else 0
        
        return combined_df, injection_point
    except Exception as e:
        print(f"加载数据失败 {datapack_folder}: {e}")
        return None, 0


def visualize_datapack(datapack_folder, output_folder):
    """可视化单个datapack的数据"""
    # 解析目标服务和故障类型
    datapack_name = os.path.basename(datapack_folder)
    target_service, fault_type = parse_service_and_fault_type(datapack_name)
    print(f"正在处理 datapack: {datapack_name}，目标服务: {target_service}，故障类型: {fault_type}")

    # 检查必要的文件是否存在
    required_paths = [
        os.path.join(datapack_folder, 'abnormal_metrics.parquet'),
        os.path.join(datapack_folder, 'normal_metrics.parquet')
    ]
    if not all(os.path.exists(path) for path in required_paths):
        print(f"缺少数据文件，跳过 {datapack_folder}")
        return

    try:
        # 加载并合并数据
        metrics_df, injection_point = load_and_combine_data(datapack_folder)
        if metrics_df is None or metrics_df.empty:
            print(f"没有有效数据，跳过 {datapack_folder}")
            return

        # 检查必要的列是否存在
        required_columns = ['metric', 'value', 'service_name']
        if not set(required_columns).issubset(metrics_df.columns):
            missing = [col for col in required_columns if col not in metrics_df.columns]
            print(f"数据中缺少必要的列: {missing}，跳过该datapack")
            return

        # 创建图表网格（指标×服务）
        fig, axes = plt.subplots(
            nrows=len(METRIC_TYPES), 
            ncols=len(COMPARE_SERVICES), 
            figsize=(4 * len(COMPARE_SERVICES), 3 * len(METRIC_TYPES))
        )
        fig.suptitle(f'{target_service} - {fault_type}', fontsize=16, y=1.02)

        # 遍历所有服务和指标组合
        for i, metric_type in enumerate(METRIC_TYPES):
            for j, service in enumerate(COMPARE_SERVICES):
                # 获取当前子图
                if len(METRIC_TYPES) > 1 and len(COMPARE_SERVICES) > 1:
                    ax = axes[i, j]
                elif len(METRIC_TYPES) > 1:
                    ax = axes[i]
                else:
                    ax = axes[j]
                
                # 筛选当前服务和指标的数据
                service_metric_data = metrics_df[
                    (metrics_df['service_name'] == service) & 
                    (metrics_df['metric'] == metric_type)
                ]
                
                # 检查是否有数据
                if service_metric_data.empty:
                    ax.text(0.5, 0.5, 'None data', ha='center', va='center')
                    ax.set_axis_off()
                    continue
                
                # 按索引排序（因为没有时间，使用默认索引）
                service_metric_data = service_metric_data.sort_index()
                
                # 绘制指标曲线 - 使用索引作为x轴
                ax.plot(service_metric_data.index, service_metric_data['value'], 'b-', linewidth=1.5)
                
                # 绘制故障注入线（正常和异常数据的分界点）
                if injection_point > 0 and injection_point < service_metric_data.index.max():
                    ax.axvline(x=injection_point, color='r', linestyle='--', linewidth=2,
                              label='Fault Injection Timestamp')
                    ax.legend(fontsize=6)

                # 设置标题和标签
                if i == 0:  # 第一行设置服务名称
                    ax.set_title(service, fontsize=8)
                if j == 0:  # 第一列设置指标类型
                    ax.set_ylabel(metric_type, fontsize=8)
                
                # 调整刻度大小
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f'{datapack_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成图表: {output_file}")

    except Exception as e:
        print(f"处理 {datapack_folder} 时出错: {e}")


def batch_visualize(root_folder, output_root):
    """批量可视化所有datapack"""
    # 查找所有datapack文件夹
    os.makedirs(output_root, exist_ok=True)
    datapack_pattern = os.path.join(root_folder, 'ts*-*')  # 匹配ts开头的datapack
    datapack_folders = [f for f in glob(datapack_pattern) if os.path.isdir(f)]
    
    print(f"找到 {len(datapack_folders)} 个datapack文件夹")
    
    # 逐个可视化
    for datapack_folder in datapack_folders:
        if any(service in os.path.basename(datapack_folder) for service in COMPARE_SERVICES):
            visualize_datapack(datapack_folder, output_root)


if __name__ == "__main__":
    # 设置根目录和输出目录
    root_directory = "__dev__rcabench_test_r1"  # 包含所有datapack的根目录
    output_directory = "study/visualize"  # 可视化结果输出目录
    
    # 执行批量可视化
    batch_visualize(root_directory, output_directory)
    print("所有可视化任务已完成！")