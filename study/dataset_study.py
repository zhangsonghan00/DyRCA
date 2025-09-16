import typer
import polars as pl
import json
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import datetime

app = typer.Typer()

def load_meta_labels(meta_dir: str, dataset: str) -> Dict[str, str]:
    """加载元数据标签，返回datapack到注入服务的映射"""
    labels_path = os.path.join(meta_dir, dataset, "labels.parquet")
    if not os.path.exists(labels_path):
        typer.echo(f"警告: 未找到元数据标签文件 {labels_path}")
        return {}
    
    # 读取labels.parquet文件
    try:
        # 尝试用polars读取parquet文件
        df = pl.read_parquet(labels_path)
        # 转换为字典: datapack -> gt.name
        return dict(zip(df["datapack"].to_list(), df["gt.name"].to_list()))
    except:
        # 如果parquet读取失败，尝试按文本行读取
        labels = {}
        with open(labels_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    datapack = data.get("datapack")
                    service = data.get("gt.name")
                    if datapack and service:
                        labels[datapack] = service
                except:
                    continue
        return labels

def process_eadro_dataset(root_dir: str, dataset: str, datapack: str) -> Tuple[str, Dict]:
    """处理eadro_tt和eadro_sn数据集"""
    datapack_path = os.path.join(root_dir, "data", dataset, datapack)
    
    # 读取fault_info.json
    fault_info_path = os.path.join(datapack_path, "fault_info.json")
    with open(fault_info_path, 'r') as f:
        fault_info = json.load(f)
    
    injection_service = fault_info["injection_name"]
    fault_type = fault_info["fault_type"]
    
    # 读取metric数据
    metric_path = os.path.join(datapack_path, "metric.parquet")
    try:
        # 尝试用polars直接读取parquet
        metric_df = pl.read_parquet(metric_path)
    except:
        # 如果失败，尝试按行读取json
        metrics = []
        with open(metric_path, 'r') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line.strip()))
                except:
                    continue
        metric_df = pl.DataFrame(metrics)
    
    # 根据故障类型确定要检查的metric
    service_points = {}
    if fault_type == "cpu_load":
        # 检查cpu_usage_total大于80的点
        cpu_metric = metric_df.filter(pl.col("metric") == "cpu_usage_total")
        # 按service_name分组并计数value > 80的点
        result = cpu_metric.with_columns(
            pl.col("value") > 80
        ).group_by("service_name").agg(
            pl.col("value").sum().alias("count")
        )
        # 转换为字典
        service_points = dict(zip(result["service_name"].to_list(), result["count"].to_list()))
    elif fault_type in ["network_loss", "network_delay"]:
        # 筛选rx_bytes和tx_bytes数据
        rx_data = metric_df.filter(pl.col("metric") == "rx_bytes").select(
            pl.col("time"), pl.col("service_name"), pl.col("value").alias("rx")
        )
        tx_data = metric_df.filter(pl.col("metric") == "tx_bytes").select(
            pl.col("time"), pl.col("service_name"), pl.col("value").alias("tx")
        )
        
        # 合并数据
        ratio_df = rx_data.join(tx_data, on=["time", "service_name"], how="inner")
        # 过滤tx为0的情况
        ratio_df = ratio_df.filter(pl.col("tx") != 0)
        # 计算比值
        ratio_df = ratio_df.with_columns(
            (pl.col("rx") / pl.col("tx")).alias("ratio")
        )
        
        # 根据数据集确定阈值范围
        if dataset == "eadro_tt":
            lower, upper = 8, 30
        else:  # eadro_sn
            lower, upper = 5, 30
            
        # 统计每个服务在范围内的点数
        ratio_df = ratio_df.with_columns(
            ((pl.col("ratio") > lower) & (pl.col("ratio") < upper)).alias("in_range")
        ).group_by("service_name").agg(
            pl.col("in_range").sum().alias("count")
        )
        
        service_points = dict(zip(ratio_df["service_name"].to_list(), ratio_df["count"].to_list()))
    
    # 分类结果
    result = classify_result(service_points, injection_service, fault_type)
    result["datapack"] = datapack
    return "eadro", result

def process_rcaeval_re2(root_dir: str, dataset: str, datapack: str) -> Tuple[str, Dict]:
    """处理rcaeval_re2_ob/tt/ss数据集"""
    datapack_path = os.path.join(root_dir, "data", dataset, datapack)
    
    # 从datapack名称提取注入服务和故障类型
    parts = datapack.split("_")
    injection_service = "_".join(parts[:-2]) if len(parts) >= 3 else parts[0]
    fault_type = parts[-2] if len(parts) >= 2 else ""
    
    # 如果是loss故障，直接返回类别（2）
    if fault_type == "loss":
        result = {
            "datapack": datapack,
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 2,
            "reason": "All services show no obvious abnormal reactions"
        }
        return "rcaeval_re2", result
    
    # 确定要检查的metric
    metric_map = {
        "cpu": "cpu",
        "mem": "mem",
        "disk": "diskio",
        "socket": "socket",
        "delay": "latency-90"
    }
    target_metric = metric_map.get(fault_type)
    if not target_metric:
        result = {
            "datapack": datapack,
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": None,
            "reason": f"不支持的故障类型: {fault_type}"
        }
        return "rcaeval_re2", result
    
    # 读取注入时间
    inject_time_path = os.path.join(datapack_path, "inject_time.txt")
    with open(inject_time_path, 'r') as f:
        inject_timestamp = int(f.read().strip())

    inject_time = datetime.datetime.fromtimestamp(inject_timestamp, datetime.UTC)
    
    # 读取metric数据
    metric_path = os.path.join(datapack_path, "simple_metrics.parquet")
    try:
        metric_df = pl.read_parquet(metric_path)
    except:
        metrics = []
        with open(metric_path, 'r') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line.strip()))
                except:
                    continue
        metric_df = pl.DataFrame(metrics)
    
    # 筛选目标metric
    target_data = metric_df.filter(pl.col("metric") == target_metric)
    
    # 区分正常和异常时期的数据
    normal_data = target_data.filter(pl.col("time") < inject_time)
    abnormal_data = target_data.filter(pl.col("time") >= inject_time)
    
    # 计算每个服务的正常时期平均值（排除0值）
    normal_avg = {}
    if not normal_data.is_empty():
        # 替换0值为null
        normal_data = normal_data.with_columns(
            pl.when(pl.col("value") == 0).then(None).otherwise(pl.col("value")).alias("value")
        )
        # 按服务分组计算平均值
        avg_df = normal_data.group_by("service_name").agg(
            pl.col("value").mean().alias("avg_value")
        ).drop_nulls()
        normal_avg = dict(zip(avg_df["service_name"].to_list(), avg_df["avg_value"].to_list()))
    
    # 计算异常时期每个点与正常平均值的比值，并统计超过阈值(3)的点数
    service_points = {}
    if not abnormal_data.is_empty() and normal_avg:

        for service, group in abnormal_data.group_by("service_name"):
            if service[0] not in normal_avg.keys():
                continue

            normal_value = normal_avg[service[0]]

            # 计算比值
            if normal_value == 0:
                # 处理正常值为0的情况
                group = group.with_columns(
                    pl.when(pl.col("value") > 0).then(float('inf')).otherwise(0).alias("ratio")
                )
            else:
                # 计算每个值与正常值的比值
                group = group.with_columns(
                    (pl.col("value") / normal_value).alias("ratio")
                )
            
            # 统计比值大于2的数量和最大比值
            count = group.filter(pl.col("ratio") > 2).height
            max_ratio = group.select(pl.col("ratio").max()).item()

            service_points[service[0]] = (count, max_ratio)

    # 分类结果
    result = classify_result_re2(service_points, injection_service, fault_type)
    result["datapack"] = datapack
    return "rcaeval_re2", result

def process_rcaeval_re3(root_dir: str, dataset: str, datapack: str) -> Tuple[str, Dict]:
    """处理rcaeval_re3_ob/tt/ss数据集（代码级故障，检查log）"""
    datapack_path = os.path.join(root_dir, "data", dataset, datapack)
    
    # 从datapack名称提取注入服务
    parts = datapack.split("_")
    injection_service = parts[0] if parts else ""
    fault_type = "_".join(parts[1:]) if len(parts) > 1 else ""
    
    # 读取注入时间
    # 读取注入时间
    inject_time_path = os.path.join(datapack_path, "inject_time.txt")
    with open(inject_time_path, 'r') as f:
        inject_timestamp = int(f.read().strip())

    inject_time = datetime.datetime.fromtimestamp(inject_timestamp, datetime.UTC)
    
    # 读取log数据
    log_path = os.path.join(datapack_path, "logs.parquet")
    try:
        log_df = pl.read_parquet(log_path)
    except:
        logs = []
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    logs.append(log_entry)
                except:
                    continue
        log_df = pl.DataFrame(logs)
    
    # # 统一时间格式为整数以便比较
    # if "time" in log_df.columns:
    #     log_df = log_df.with_columns(
    #         pl.col("time").cast(pl.Float64).fill_null(0)
    #         .map_elements(lambda x: int(str(x).replace(".", "")[:13]), return_dtype=pl.Int64)
    #         .alias("time")
    #     )
    
    # 筛选有错误的日志
    error_logs = log_df.filter(pl.col("attr.has_error") == True)
    
    # 区分正常和异常时期的错误日志
    normal_errors = error_logs.filter(pl.col("time") < inject_time)
    abnormal_errors = error_logs.filter(pl.col("time") >= inject_time)
    
    # 统计正常时期每个服务的错误数量
    normal_counts = {}
    if not normal_errors.is_empty():
        count_df = normal_errors.group_by("service_name").agg(pl.len().alias("count"))
        normal_counts = dict(zip(count_df["service_name"].to_list(), count_df["count"].to_list()))
    
    # 统计异常时期每个服务的错误数量
    abnormal_counts = {}
    if not abnormal_errors.is_empty():
        count_df = abnormal_errors.group_by("service_name").agg(pl.len().alias("count"))
        abnormal_counts = dict(zip(count_df["service_name"].to_list(), count_df["count"].to_list()))
    
    # 计算错误数量比率
    error_ratios = {}
    for service in abnormal_counts.keys():

        normal_count = normal_counts.get(service, 0)
        abnormal_count = abnormal_counts[service]
        
        if normal_count == 0:
            # error_ratios[service] = float('inf') if abnormal_count > 0 else 0
            error_ratios[service] = abnormal_count / 1
        else:
            error_ratios[service] = abnormal_count / normal_count
    
    
    # 分类结果
    result = classify_result_re3(
        normal_counts, abnormal_counts, error_ratios, 
        injection_service, fault_type
    )

    result["datapack"] = datapack
    return "rcaeval_re3", result

def process_aiops21(root_dir: str, dataset: str, datapack: str) -> Tuple[str, Dict]:
    """处理aiops21数据集"""
    datapack_path = os.path.join(root_dir, "data", dataset, datapack)
    
    # 读取metadata.json获取关键信息
    metadata_path = os.path.join(datapack_path, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    injection_service = metadata["injection_name"]
    fault_type = metadata["fault_type"]
    
    # 解析时间（转换为UTC时间戳）
    def parse_time(time_str):
        return datetime.datetime.fromisoformat(time_str)  
    
    normal_start_time = parse_time(metadata["normal_start_time"])
    normal_end_time = parse_time(metadata["normal_end_time"])
    fault_start_time = parse_time(metadata["fault_start_time"])
    fault_end_time = parse_time(metadata["fault_end_time"])
    
    # 根据故障类型确定要检查的指标和处理逻辑
    if fault_type in ["NETWORK_delay", "NETWORK_loss"]:
        # 处理网络相关故障，使用traces.parquet计算latency
        result = process_network_fault(
            datapack_path, injection_service, fault_type,
            normal_start_time, normal_end_time,
            fault_start_time, fault_end_time
        )
    else:
        # 处理其他故障，使用metrics.parquet
        result = process_metric_based_fault(
            datapack_path, injection_service, fault_type,
            normal_start_time, normal_end_time,
            fault_start_time, fault_end_time
        )
    
    result["datapack"] = datapack
    return "aiops21", result

def process_metric_based_fault(datapack_path: str, injection_service: str, fault_type: str,
                              normal_start: datetime.datetime, normal_end: datetime.datetime,
                              fault_start: datetime.datetime, fault_end: datetime.datetime) -> Dict:
    """处理基于metric的故障类型"""
    # 定义故障类型与指标的映射关系
    metric_mappings = {
        "CPU_stress": ["OSLinux-CPU_CPU_CPUCpuUtil"],
        "JVMCPU_stress": ["JVM-Operating System_7779_JVM_JVM_CPULoad", 
                         "JVM-Operating System_7778_JVM_JVM_CPULoad"],
        "DISK_payload": ["OSLinux-OSLinux_LOCALDISK_LOCALDISK-sda_DSKPercentBusy",
                        "OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKPercentBusy"],
        "DISK_usage": ["OSLinux-OSLinux_LOCALDISK_LOCALDISK-sda_DSKPercentBusy",
                      "OSLinux-OSLinux_LOCALDISK_LOCALDISK-sdb_DSKPercentBusy"],
        "JVMMEMORY_OOM": ["JVM-Memory_7778_JVM_Memory_HeapMemoryUsage",
                         "JVM-Memory_7779_JVM_Memory_HeapMemoryUsage"],
        "MEMORY_stress": ["OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc"]
    }
    
    # 获取当前故障类型需要关注的指标
    target_metrics = metric_mappings.get(fault_type)
    if not target_metrics:
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": None,
            "reason": f"不支持的故障类型: {fault_type}"
        }
    
    # 读取metric数据
    metric_path = os.path.join(datapack_path, "metrics.parquet")
    try:
        metric_df = pl.read_parquet(metric_path)
    except:
        metrics = []
        with open(metric_path, 'r') as f:
            for line in f:
                try:
                    metrics.append(json.loads(line.strip()))
                except:
                    continue
        metric_df = pl.DataFrame(metrics)
    
    # 筛选在正常时间范围内的目标指标数据
    normal_data = metric_df.filter(
        (pl.col("time") >= normal_start) & 
        (pl.col("time") <= normal_end) &
        (pl.col("metric").is_in(target_metrics))
    )
    
    # 筛选在故障时间范围内的目标指标数据
    abnormal_data = metric_df.filter(
        (pl.col("time") >= fault_start) & 
        (pl.col("time") <= fault_end) &
        (pl.col("metric").is_in(target_metrics))
    )
    
    # 计算每个服务的正常时期平均值（排除0值）
    normal_avg = {}
    if not normal_data.is_empty():
        # 替换0值为null
        normal_data = normal_data.with_columns(
            pl.when(pl.col("value") == 0).then(None).otherwise(pl.col("value")).alias("value")
        )
        # 按服务和指标分组计算平均值，然后合并同一服务的多个指标
        avg_df = normal_data.group_by("service_name", "metric").agg(
            pl.col("value").mean().alias("avg_value")
        ).drop_nulls()
        
        # 合并同一服务的多个指标平均值（取平均）
        for service, group in avg_df.group_by("service_name"):
            normal_avg[service] = group["avg_value"].mean()
    
    # 计算异常时期每个点与正常平均值的比值，并统计超过阈值(3)的点数
    service_points = {}
    if not abnormal_data.is_empty() and normal_avg:
        # 处理多个指标的情况，合并计数
        metric_counts = {}
        
        for metric in target_metrics:
            metric_data = abnormal_data.filter(pl.col("metric") == metric)
            
            for service, group in metric_data.group_by("service_name"):
                if service not in normal_avg:
                    continue
                
                normal_value = normal_avg[service]
                
                # 计算比值
                if normal_value == 0:
                    group = group.with_columns(
                        pl.when(pl.col("value") > 0).then(float('inf')).otherwise(0).alias("ratio")
                    )
                else:
                    group = group.with_columns(
                        (pl.col("value") / normal_value).alias("ratio")
                    )

                # 统计比值大于2的数量和最大比值
                count = group.filter(pl.col("ratio") > 3).height
                max_ratio = group.select(pl.col("ratio").max()).item()
                
                # 合并多个指标的计数
                if service not in metric_counts:
                    metric_counts[service] = (count, max_ratio)
                else:
                    curr_count, curr_max = metric_counts[service]
                    metric_counts[service] = (curr_count + count, max(max_ratio, curr_max))
        
        service_points = metric_counts
    
    # 分类结果
    return classify_result_re2(service_points, injection_service, fault_type)

def process_network_fault(datapack_path: str, injection_service: str, fault_type: str,
                         normal_start: datetime.datetime, normal_end: datetime.datetime,
                         fault_start: datetime.datetime, fault_end: datetime.datetime) -> Dict:
    """处理网络相关故障（基于traces计算latency）"""
    # 读取trace数据
    trace_path = os.path.join(datapack_path, "traces.parquet")
    try:
        trace_df = pl.read_parquet(trace_path)
    except:
        traces = []
        with open(trace_path, 'r') as f:
            for line in f:
                try:
                    trace = json.loads(line.strip())
                    traces.append(trace)
                except:
                    continue
        trace_df = pl.DataFrame(traces)
    
    # 确保duration列存在
    if "duration" not in trace_df.columns:
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": None,
            "reason": "traces数据中缺少duration列"
        }
    
    # 筛选正常时期的trace数据
    normal_traces = trace_df.filter(
        (pl.col("time") >= normal_start) & 
        (pl.col("time") <= normal_end)
    )
    
    # 筛选故障时期的trace数据
    abnormal_traces = trace_df.filter(
        (pl.col("time") >= fault_start) & 
        (pl.col("time") <= fault_end)
    )
    
    # 计算正常时期每个服务的p90 latency
    normal_p90 = {}
    if not normal_traces.is_empty():
        # 按服务分组计算p90 latency
        p90_df = normal_traces.group_by("service_name").agg(
            pl.col("duration").quantile(0.9).alias("p90_latency")
        ).drop_nulls()
        normal_p90 = dict(zip(p90_df["service_name"].to_list(), p90_df["p90_latency"].to_list()))
    
    # 计算故障时期每个服务的p90 latency与正常时期的比值
    service_points = {}
    if not abnormal_traces.is_empty() and normal_p90:
        # 按服务分组计算p90 latency
        abnormal_p90_df = abnormal_traces.group_by("service_name").agg(
            pl.col("duration").quantile(0.9).alias("p90_latency")
        ).drop_nulls()
        
        # 计算比值并统计
        for service, p90 in zip(abnormal_p90_df["service_name"].to_list(), 
                               abnormal_p90_df["p90_latency"].to_list()):
            if service not in normal_p90:
                continue
                
            normal_p90_val = normal_p90[service]
            
            # 计算比值
            if normal_p90_val == 0:
                ratio = float('inf') if p90 > 0 else 0
            else:
                ratio = p90 / normal_p90_val

            # 对于网络故障，使用阈值3
            count = 1 if ratio > 3 else 0
            service_points[service] = (count, ratio)
    
    # 分类结果
    return classify_result_re2(service_points, injection_service, fault_type)

def classify_result(service_points: Dict[str, int], injection_service: str, fault_type: str) -> Dict:
    """对结果进行分类"""
    # 如果没有任何服务有异常点，归类为（2）
    if not service_points or all(count == 0 for count in service_points.values()):
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 2,
            "reason": "All services show no obvious abnormal reactions"
        }
    
    # 获取注入服务的点数
    injection_count = service_points.get(injection_service, 0)
    
    # 检查是否有其他服务的点数高于注入服务
    other_services = {s: c for s, c in service_points.items() if s != injection_service}
    has_higher_other = any(count > injection_count for count in other_services.values())
    
    if has_higher_other:
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 3,
            "reason": "Other services have stronger fault reactions"
        }
    elif injection_count > 0:
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 1,
            "reason": "Only the injected service has an obvious abnormal reaction"
        }
    else:
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 3,
            "reason": "注入服务无反应，但其他服务有反应"
        }

def classify_result_re2(service_points: Dict[str, Tuple[int, float]], injection_service: str, fault_type: str) -> Dict:
    """为rcaeval_re2数据集分类结果，考虑点数相同的情况"""
    # 如果没有任何服务有异常点，归类为（2）
    if not service_points or all(count == 0 for count, _ in service_points.values()):
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 2,
            "reason": "All services show no obvious abnormal reactions"
        }
    
    # 获取注入服务的点数和最大比值
    injection_data = service_points.get(injection_service, (0, 0))
    injection_count, injection_max_ratio = injection_data
    
    # 检查是否有其他服务的点数高于注入服务，或点数相同但最大比值更高
    other_services = {s: (c, r) for s, (c, r) in service_points.items() if s != injection_service}
    has_better_other = False
    
    for _, (count, ratio) in other_services.items():
        if count > injection_count or (count == injection_count and ratio > injection_max_ratio):
            has_better_other = True
            break
    
    if has_better_other:
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 3,
            "reason": "Other services have stronger fault reactions"
        }
    elif injection_count > 0:
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 1,
            "reason": "Only the injected service has an obvious abnormal reaction"
        }
    else:
        return {
            "injection_service": injection_service,
            "fault_type": fault_type,
            "classification": 3,
            "reason": "Other services have stronger fault reactions"
        }

def classify_result_re3(normal_counts: Dict[str, int], abnormal_counts: Dict[str, int], 
                       error_ratios: Dict[str, float], injection_service: str, fault_type: str) -> Dict:
    """为rcaeval_re3数据集分类结果（基于日志错误）"""
    # 检查正常时期是否有错误
    has_normal_errors = any(count > 0 for count in normal_counts.values())
    
    injection_abnormal = abnormal_counts.get(injection_service, 0)
    injection_ratio = error_ratios.get(injection_service, 0)

    # 检查注入服务是否在异常数量或比率上最高
    is_injection_highest = True
    
    for service in abnormal_counts:
        if service == injection_service:
            continue
            
        # 比较异常数量
        if abnormal_counts[service] > injection_abnormal:
            is_injection_highest = False
            break
            
        # 如果异常数量相同，比较比率
        if abnormal_counts[service] == injection_abnormal and error_ratios[service] > injection_ratio:
            is_injection_highest = False
            break
    
    # 如果正常时期没有错误
    if not has_normal_errors:
        # 检查是否有任何服务有异常错误
        has_abnormal_errors = any(count > 0 for count in abnormal_counts.values())
        
        if not has_abnormal_errors:
            return {
                "injection_service": injection_service,
                "fault_type": fault_type,
                "classification": 2,
                "reason": "All services show no obvious abnormal reactions"
            }
        elif is_injection_highest and injection_abnormal > 0:
            return {
                "injection_service": injection_service,
                "fault_type": fault_type,
                "classification": 1,
                "reason": "Only the injected service has an obvious abnormal reaction"
            }
        else:
            return {
                "injection_service": injection_service,
                "fault_type": fault_type,
                "classification": 3,
                "reason": "Other services have stronger fault reactions"
            }
    else:
        # 正常时期有错误，使用比率和异常数量进行判断
        if is_injection_highest and (injection_abnormal > 0 or injection_ratio > 1):
            return {
                "injection_service": injection_service,
                "fault_type": fault_type,
                "classification": 1,
                "reason": "注入服务在错误数量或比率上最高"
            }
        elif all(count == 0 for count in abnormal_counts.values()):
            return {
                "injection_service": injection_service,
                "fault_type": fault_type,
                "classification": 2,
                "reason": "无异常错误日志"
            }
        else:
            return {
                "injection_service": injection_service,
                "fault_type": fault_type,
                "classification": 3,
                "reason": "其他服务在错误数量或比率上更高"
            }

@app.command()
def telemetry_analyze(
    root_dir: str = typer.Option("data/rcabench-platform-v2", help="数据根目录路径"),
    dataset: str = typer.Option(..., help="数据集名称"),
    output: str = typer.Option("output/rcabench-platform-v2/study", help="批量分析结果输出文件路径")
):
    """分析数据集的遥测数据，判断故障模式是否简单"""
    typer.echo(f"开始分析数据集: {dataset}")
    typer.echo(f"数据根目录: {root_dir}")
    
    # 创建输出目录
    Path(output).mkdir(parents=True, exist_ok=True)
    
    # 加载元数据标签
    meta_dir = os.path.join(root_dir, "meta")
    datapack_services = load_meta_labels(meta_dir, dataset)
    
    # 获取数据集中的所有datapack
    data_dir = os.path.join(root_dir, "data", dataset)
    if not os.path.exists(data_dir):
        typer.echo(f"错误: 数据目录 {data_dir} 不存在")
        return
    
    datapacks = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    datapacks = [d for d in datapacks if d in datapack_services.keys()]
    if not datapacks:
        typer.echo(f"警告: 在 {data_dir} 中未找到任何datapack")
        return
    
    # 处理每个datapack
    results = []
    for datapack in datapacks:
        # typer.echo(f"处理 datapack: {datapack}")
        
        try:
            # 根据数据集类型选择不同的处理函数
            if dataset in ["eadro_tt", "eadro_sn"]:
                _, result = process_eadro_dataset(root_dir, dataset, datapack)
            elif dataset in ["rcaeval_re2_ob", "rcaeval_re2_tt", "rcaeval_re2_ss"]:
                _, result = process_rcaeval_re2(root_dir, dataset, datapack)
            elif dataset in ["rcaeval_re3_ob", "rcaeval_re3_tt", "rcaeval_re3_ss"]:
                _, result = process_rcaeval_re3(root_dir, dataset, datapack)
            elif dataset == "aiops21":  # 添加aiops21数据集处理
                _, result = process_aiops21(root_dir, dataset, datapack)
            else:
                typer.echo(f"警告: 不支持的数据集 {dataset}")
                continue
            
            results.append(result)
            
            # 输出分类结果
            # typer.echo(f"  Classification result: Category（{result['classification']}） - {result['reason']}")
            
        except Exception as e:
            typer.echo(f"处理 datapack {datapack} 时出错: {str(e)}")
            continue
    
    # 保存结果到输出文件
    output_file = os.path.join(output, f"{dataset}_classification_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"分析完成，结果已保存到 {output_file}")

    # 重排结果，对于rcaeval3/aiops21(根据simplerca结果)
    if dataset in ["rcaeval_re3_ob", "rcaeval_re3_tt", "rcaeval_re3_ss", "aiops21"]:
        meta_file="output/rcabench-platform-v2/stats/min_rank_statistics.json"
        with open(meta_file, 'r') as f:
            meta_data= json.load(f)
        simplerca_results=meta_data.get(dataset,[])
        not_top1_datapacks=simplerca_results["min_rank_not_top1_datapacks"]
        new_results=[]
        for res in results:
            if res["datapack"] not in not_top1_datapacks:
                res["classification"]=1
                res["reason"]="Only the injected service has an obvious abnormal reaction"
            new_results.append(res)
        results=new_results

    # 统计各类别的数量
    class_counts = {}
    for res in results:
        cls = res["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    typer.echo("分类统计:")
    for cls, count in class_counts.items():
        typer.echo(f"  类别（{cls}）: {round(count/len(datapacks), 2)} 个case")

if __name__ == "__main__":
    app()