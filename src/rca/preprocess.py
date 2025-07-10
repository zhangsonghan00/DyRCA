import pandas as pd
import numpy as np
import torch
import dgl
import pickle
import json
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
import os
from rcabench.openapi import InjectionApi, ApiClient, Configuration

import statistics  # 用于trace latency std 计算

from enum import Enum, auto  # 导入Enum基类和auto值生成器

class Dataset(Enum):  # 定义名为Dataset的枚举类，继承自Enum
    RCABENCH_database = auto()
    RCABENCH_filtered = auto()  
    OTHER = auto()


# 配置参数
SAMPLING_SIZE = 5  # 数据平滑粒度为5秒
WINDOW_SIZE = 10  # 滑动窗口大小为10个采样点

def create_dataset(
    data_root: str,
    output_dir: str,
    max_cases: Optional[int] = None,
    train_ratio: float = 0.7,
    enable_checkpointing: bool = True,
    ds: str = "rcabench",
) -> List[Path]:
    if ds == "rcabench":
        dataset = Dataset.RCABENCH_database
    elif ds == "rcabench_filtered":
        dataset = Dataset.RCABENCH_filtered
    elif ds == "other":
        dataset = Dataset.OTHER


    logger.info("Starting dataset creation...")
    output_path = Path(output_dir) / ds

    if dataset == Dataset.RCABENCH_database:
        config = Configuration(host="http://10.10.10.220:32080")
        with ApiClient(configuration=config) as client:
            api = InjectionApi(api_client=client)
            resp = api.api_v1_injections_analysis_with_issues_get()

        assert resp.data is not None, "No cases found in the response"
        case_names = list(
            set([item.injection_name for item in resp.data if item.injection_name])
        )
        data_packs = [Path(data_root) / name / "converted" for name in case_names]

        data_packs = data_packs[:max_cases]

    elif dataset == Dataset.RCABENCH_filtered:
        logger.info(f"Using filtered dataset from {data_root}")
        service_select = ["preserve-service", "security-service", "order-service", 
                        "consign-service", "consign-price-service","food-service",
                        "station-food-service","train-food-service","assurance-service",
                        "preserve-other-service","order-other-service","contact-service",
                        "station-service","seat-service"]

        service_path_name =[]

        # 遍历目录
        for root, dirs, files in os.walk(data_root):
            for dir_name in dirs:
                if any(sample in dir_name for sample in service_select):
                    service_path_name.append(dir_name)

        data_packs = [Path(data_root) / name for name in service_path_name]

    return data_packs


class DataPreprocessor:
    def __init__(self, metrics_path: str, traces_path: str, logs_path: str, 
                 all_services: List[str], all_metrics: List[str], sampling_size: int = SAMPLING_SIZE):
        self.metrics_path = metrics_path
        self.traces_path = traces_path
        self.logs_path = logs_path
        self.sampling_size = sampling_size
        self.metrics_df: Optional[pd.DataFrame] = None
        self.traces_df: Optional[pd.DataFrame] = None
        self.logs_df: Optional[pd.DataFrame] = None
        self.services = all_services
        self.metrics = all_metrics
        self.service_to_idx = {service: i for i, service in enumerate(all_services)}
        self.metric_to_idx = {metric: i for i, metric in enumerate(all_metrics)}
        
        # 存储处理后的数据
        self.features_by_window: Optional[Dict[int, torch.Tensor]] = None
        self.trace_edges_by_window: Optional[Dict[int, List[Dict]]] = None  # 存储原始边信息
        self.time_windows: Optional[List[int]] = None

    @staticmethod
    def extract_service_name(row: pd.Series) -> Optional[str]:
        if row['service_name'] != "":
            return row['service_name']
        elif 'attr.k8s.container.name' in row and row['attr.k8s.container.name'] != "":
            return row['attr.k8s.container.name']
        elif 'attr.k8s.service.name' in row and row['attr.k8s.service.name'] != "":
            return row['attr.k8s.service.name']
        return None

    def load_data(self) -> None:
        self.metrics_df = pd.read_parquet(self.metrics_path)
        self.metrics_df['value'] = self.metrics_df['value'].fillna(0)
        self.traces_df = pd.read_parquet(self.traces_path)
        self.logs_df = pd.read_parquet(self.logs_path)

    def preprocess_data(self) -> None:
        """统一处理metrics、traces、logs数据，提取特征并构建图"""
        if any(df is None for df in [self.metrics_df, self.traces_df, self.logs_df]):
            raise ValueError("数据未加载，请先调用 load_data() 方法")
        
        # 1. 处理metrics数据
        metrics_features = self._process_metrics()
        
        # 2. 处理traces数据，只提取特征和边信息，不构建图
        trace_features, trace_edges = self._process_traces()
        
        # 3. 处理logs数据
        log_features = self._process_logs()
        
        # 4. 合并所有特征
        self._combine_features(metrics_features, trace_features, log_features)
        
        # 5. 保存边信息用于后续图构建
        self.trace_edges_by_window = trace_edges
        
        if self.time_windows:
            logger.info(f"处理完成，共 {len(self.time_windows)} 个时间窗口")

    def _process_metrics(self) -> Dict[int, torch.Tensor]:
        """处理metrics数据"""
        if self.metrics_df is None:
            return {}
        
        df = self.metrics_df.copy()
        df['extracted_service'] = df.apply(DataPreprocessor.extract_service_name, axis=1)  # type: ignore
        df = df.dropna(subset=['extracted_service']).copy()
        df.loc[:, 'timestamp'] = df['time'].astype('int64') // 1_000_000_000
        df.loc[:, 'time_window'] = df['timestamp'] // self.sampling_size
        
        # 按时间窗口、服务、指标聚合
        grouped = df.groupby(['time_window', 'extracted_service', 'metric'])['value'].mean().reset_index()
        
        # 构建每个时间窗口的特征矩阵，使用全局metrics集合
        features_by_window = {}
        time_windows = sorted(grouped['time_window'].unique())
        
        for window in time_windows:
            window_data = grouped[grouped['time_window'] == window]
            # 使用全局metrics的数量创建特征矩阵
            feature_matrix = torch.zeros((len(self.services), len(self.metrics)))
            
            for _, row in window_data.iterrows():
                service_idx = self.service_to_idx.get(row['extracted_service'])
                metric_idx = self.metric_to_idx.get(row['metric'])
                if service_idx is not None and metric_idx is not None:
                    feature_matrix[service_idx, metric_idx] = row['value']
            
            features_by_window[window] = feature_matrix
        
        return features_by_window

    def _process_traces(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[Dict]]]:
        """处理traces数据，返回特征和边信息，并提取trace latency特征"""
        if self.traces_df is None:
            return {}, {}
        df = self.traces_df.copy()
        df['extracted_service'] = df.apply(DataPreprocessor.extract_service_name, axis=1)  # type: ignore
        df = df.dropna(subset=['extracted_service']).copy()
        df.loc[:, 'timestamp'] = df['time'].astype('int64') // 1_000_000_000
        df.loc[:, 'time_window'] = df['timestamp'] // self.sampling_size
        # 构建服务调用关系
        df['source_service'] = df['extracted_service']
        df['destination_service'] = None
        span_service_map = df.set_index('span_id')['extracted_service'].to_dict()
        def find_destination(row) -> Optional[str]:
            parent_span_id = row['parent_span_id']
            if parent_span_id in span_service_map:
                return span_service_map[parent_span_id]
            return None
        df['destination_service'] = df.apply(find_destination, axis=1)  # type: ignore
        df = df.dropna(subset=['source_service', 'destination_service']).copy()
        df['is_error'] = df['attr.status_code'].isin(['Error'])
        trace_features = {}
        trace_edges = {}
        time_windows = sorted(df['time_window'].unique())
        for window in time_windows:
            window_data = df[df['time_window'] == window]
            # 1. 提取trace特征 (duration, request_count, latency_p90, latency_std)
            trace_stats = window_data.groupby('extracted_service').agg({
                'duration': 'mean',
                'trace_id': 'count'
            }).reset_index()
            trace_stats.columns = ['service', 'avg_duration', 'request_count']
            # === 新增trace latency特征 ===
            latency_p90_list = []
            latency_std_list = []
            for service in self.services:
                service_spans = window_data[window_data['extracted_service'] == service]
                parent_span_df = window_data.set_index('span_id')
                latency_list = []
                for _, span in service_spans.iterrows():
                    parent_id = span['parent_span_id']
                    span_time = span['time']
                    span_duration = span['duration']
                    if parent_id is None or pd.isna(parent_id) or parent_id == "":
                        continue
                    try:
                        if parent_id in parent_span_df.index:
                            parent_span = parent_span_df.loc[parent_id]
                            if isinstance(parent_span, pd.Series):
                                if parent_span['extracted_service'] != service:
                                    latency = span_duration / 1000
                                    latency_list.append(latency)
                            else:
                                for _, p_span in parent_span.iterrows():
                                    if p_span['extracted_service'] != service:
                                        latency = span_duration / 1000
                                        latency_list.append(latency)
                    except Exception:
                        continue
                if len(latency_list) > 2:
                    latency_p90 = np.percentile(latency_list, 90)
                    latency_std = statistics.stdev(latency_list)
                else:
                    latency_p90 = 1
                    latency_std = 1
                latency_p90_list.append(latency_p90)
                latency_std_list.append(latency_std)
            # === END ===
            feature_matrix = torch.zeros((len(self.services), 4))  # duration, count, latency_p90, latency_std
            for idx, row in trace_stats.iterrows():
                service_idx = self.service_to_idx.get(row['service'])
                if service_idx is not None:
                    feature_matrix[service_idx, 0] = row['avg_duration']
                    feature_matrix[service_idx, 1] = row['request_count']
            for i, (lat_p90, lat_std) in enumerate(zip(latency_p90_list, latency_std_list)):
                feature_matrix[i, 2] = lat_p90
                feature_matrix[i, 3] = lat_std
            trace_features[window] = feature_matrix
            
            # 2. 收集边信息（不立即构建图）
            edge_stats = window_data.groupby(['source_service', 'destination_service']).agg({
                'trace_id': 'count',  # 调用频率
                'is_error': 'mean'    # 错误率
            }).reset_index()
            edge_stats.columns = ['src_service', 'dst_service', 'call_frequency', 'error_rate']
            
            edge_list = []
            for _, row in edge_stats.iterrows():
                src_idx = self.service_to_idx.get(row['src_service'])
                dst_idx = self.service_to_idx.get(row['dst_service'])
                
                if src_idx is not None and dst_idx is not None and src_idx != dst_idx:
                    edge_list.append({
                        'src': src_idx,
                        'dst': dst_idx,
                        'call_frequency': row['call_frequency'],
                        'error_rate': row['error_rate']
                    })
            
            trace_edges[window] = edge_list
        
        return trace_features, trace_edges

    def _process_logs(self) -> Dict[int, torch.Tensor]:
        """处理logs数据"""
        if self.logs_df is None:
            return {}
        
        df = self.logs_df.copy()
        df['extracted_service'] = df.apply(DataPreprocessor.extract_service_name, axis=1)  # type: ignore
        df = df.dropna(subset=['extracted_service']).copy()
        df.loc[:, 'timestamp'] = df['time'].astype('int64') // 1_000_000_000
        df.loc[:, 'time_window'] = df['timestamp'] // self.sampling_size
        
        # 统计各级别日志数量
        log_levels = ['INFO', 'ERROR', 'WARN', 'DEBUG']
        log_counts = (
            df.groupby(['time_window', 'extracted_service', 'level'])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        
        # 确保所有级别都存在
        for level in log_levels:
            if level not in log_counts.columns:
                log_counts[level] = 0
        
        # 构建每个时间窗口的日志特征
        features_by_window = {}
        time_windows = sorted(log_counts['time_window'].unique())
        
        for window in time_windows:
            window_data = log_counts[log_counts['time_window'] == window]
            feature_matrix = torch.zeros((len(self.services), len(log_levels)))
            
            for _, row in window_data.iterrows():
                service_idx = self.service_to_idx.get(row['extracted_service'])
                if service_idx is not None:
                    for i, level in enumerate(log_levels):
                        feature_matrix[service_idx, i] = row[level]
            
            features_by_window[window] = feature_matrix
        
        return features_by_window

    def _combine_features(self, metrics_features: Dict[int, torch.Tensor], 
                         trace_features: Dict[int, torch.Tensor], 
                         log_features: Dict[int, torch.Tensor]) -> None:
        """合并所有特征"""
        # 获取所有时间窗口
        all_windows = set(metrics_features.keys()) | set(trace_features.keys()) | set(log_features.keys())
        self.time_windows = sorted(all_windows)
        
        combined_features = {}
        for window in self.time_windows:
            feature_list = []
            
            # 添加metrics特征
            if window in metrics_features:
                feature_list.append(metrics_features[window])
            else:
                # 如果该窗口没有metrics数据，用零填充
                feature_list.append(torch.zeros((len(self.services), len(self.metrics))))
            
            # 添加trace特征
            if window in trace_features:
                feature_list.append(trace_features[window])
            else:
                feature_list.append(torch.zeros((len(self.services), 4)))
            
            # 添加log特征
            if window in log_features:
                feature_list.append(log_features[window])
            else:
                feature_list.append(torch.zeros((len(self.services), 4)))
            
            # 拼接所有特征
            combined_features[window] = torch.cat(feature_list, dim=1)
        
        self.features_by_window = combined_features

    def combine_data(self) -> Tuple[List[int], List[torch.Tensor], List[List[Dict]]]:
        """返回处理后的数据用于滑动窗口采样"""
        if self.features_by_window is None or self.trace_edges_by_window is None or self.time_windows is None:
            raise ValueError("数据未处理，请先调用 preprocess_data() 方法")
        
        timestamps = []
        features = []
        edges = []
        
        for window in self.time_windows:
            timestamps.append(window)
            features.append(self.features_by_window[window])
            edges.append(self.trace_edges_by_window.get(window, []))
        
        return timestamps, features, edges

class DatasetPack:
    """每个case的数据包，包含滑动窗口样本提取功能"""
    def __init__(self, timestamps: List[int], features: List[torch.Tensor], edges: List[List[Dict]]):
        self.timestamps = timestamps
        self.features = features
        self.edges = edges  # 每个时间窗口的边信息列表

    def create_sliding_windows(self, window_size: int = WINDOW_SIZE, max_gap: int = 15, 
                             all_services: Optional[List[str]] = None) -> List[Tuple[torch.Tensor, dgl.DGLGraph]]:
        """创建滑动窗口样本，基于整个滑动窗口构建图"""
        if len(self.features) < window_size:
            return []
        
        if all_services is None:
            all_services = [f"service_{i}" for i in range(self.features[0].shape[0])]
        
        service_to_idx = {service: i for i, service in enumerate(all_services)}
        
        samples = []
        for i in range(len(self.features) - window_size + 1):
            # 检查时间间隔是否合理
            time_diffs = [abs(self.timestamps[i+j+1] - self.timestamps[i+j]) 
                         for j in range(window_size-1)]
            
            if all(diff <= max_gap for diff in time_diffs):
                # 特征序列 (window_size, num_nodes, num_features)
                feature_sequence = torch.stack(self.features[i:i+window_size])
                
                # 基于整个滑动窗口构建图
                window_graph = self._build_graph_from_window(
                    self.edges[i:i+window_size], 
                    feature_sequence,
                    len(all_services)
                )
                
                samples.append((feature_sequence, window_graph))
        
        return samples
    
    def _build_graph_from_window(self, window_edges: List[List[Dict]], 
                                window_features: torch.Tensor,
                                num_nodes: int) -> dgl.DGLGraph:
        """基于滑动窗口中的所有边信息和特征构建综合图"""
        
        # 统计整个窗口中的边连接信息
        edge_stats = defaultdict(lambda: {'call_frequencies': [], 'error_rates': []})
        
        # 收集窗口内所有边信息
        for edge_list in window_edges:
            for edge in edge_list:
                key = (edge['src'], edge['dst'])
                edge_stats[key]['call_frequencies'].append(edge['call_frequency'])
                edge_stats[key]['error_rates'].append(edge['error_rate'])
        
        # 构建图
        graph = dgl.graph(([], []), num_nodes=num_nodes)
        
        if edge_stats:
            src_list = []
            dst_list = []
            weight_list = []
            
            # 收集所有边的权重用于标准化
            all_call_freqs = []
            all_error_rates = []
            
            for (src, dst), stats in edge_stats.items():
                avg_call_freq = np.mean(stats['call_frequencies'])
                avg_error_rate = np.mean(stats['error_rates'])
                all_call_freqs.append(avg_call_freq)
                all_error_rates.append(avg_error_rate)
            
            # 标准化
            if all_call_freqs:
                call_freq_array = np.array(all_call_freqs)
                error_rate_array = np.array(all_error_rates)
                
                # Min-Max标准化
                def minmax_normalize_np(x):
                    min_val = x.min()
                    max_val = x.max()
                    if max_val - min_val < 1e-8:
                        return np.zeros_like(x)
                    return (x - min_val) / (max_val - min_val)
                
                norm_call_freqs = minmax_normalize_np(call_freq_array)
                norm_error_rates = minmax_normalize_np(error_rate_array)
                
                # 构建边
                for idx, ((src, dst), _) in enumerate(edge_stats.items()):
                    src_list.append(src)
                    dst_list.append(dst)
                    # 加权组合
                    weight = 0.5 * norm_call_freqs[idx] + 0.5 * norm_error_rates[idx]
                    weight_list.append(weight)
                
                graph.add_edges(src_list, dst_list)
                graph.edata['weight'] = torch.tensor(weight_list, dtype=torch.float32)
        
        return graph

def extract_injection_name(injection_name: str) -> str:
    parts = injection_name.split('-')
    for i, p in enumerate(parts):
        if p == 'service' and i > 0:
            return '-'.join(parts[:i+1])
    return injection_name

def minmax_normalize_features(feature_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """对特征列表进行Min-Max标准化"""
    if not feature_list:
        return []
    
    # 将所有特征拼接
    all_features = torch.stack(feature_list)  # (num_samples, num_nodes, num_features)
    
    # 计算每个特征维度的min和max
    min_vals = all_features.min(dim=0, keepdim=True)[0].min(dim=0, keepdim=True)[0]  # (1, 1, num_features)
    max_vals = all_features.max(dim=0, keepdim=True)[0].max(dim=0, keepdim=True)[0]  # (1, 1, num_features)
    
    # 避免分母为0
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-8] = 1.0
    
    # 标准化
    normalized = (all_features - min_vals) / range_vals
    
    return [normalized[i] for i in range(normalized.shape[0])]

def collect_all_services_and_metrics(case_dirs: List[Path], metric_file_name: str) -> Tuple[List[str], List[str]]:
    """收集所有服务名称和指标名称"""
    all_services = set()
    all_metrics = set()
    
    for case_dir in case_dirs:
        metric_path = case_dir / metric_file_name
        if not metric_path.exists():
            continue
        df = pd.read_parquet(metric_path)
        df['extracted_service'] = df.apply(DataPreprocessor.extract_service_name, axis=1)  # type: ignore
        
        # 收集服务
        services = df['extracted_service'].dropna().unique()
        all_services.update(services)
        
        # 收集指标
        metrics = df['metric'].unique()
        all_metrics.update(metrics)
        
        print(f"Case {case_dir.name}: {len(services)} 个服务, {len(metrics)} 个指标")
    
    return sorted(all_services), sorted(all_metrics)


def process_case(case_dir: Path, metric_file: str, trace_file: str, log_file: str, 
                all_services: List[str], all_metrics: List[str], sampling_size: int = SAMPLING_SIZE) -> Tuple[List[int], List[torch.Tensor], List[List[Dict]]]:
    """处理单个case，返回时间戳、特征和边信息"""
    processor = DataPreprocessor(
        metrics_path=os.path.join(case_dir, metric_file),
        traces_path=os.path.join(case_dir, trace_file),
        logs_path=os.path.join(case_dir, log_file),
        all_services=all_services,
        all_metrics=all_metrics,
        sampling_size=sampling_size
    )
    processor.load_data()
    processor.preprocess_data()
    return processor.combine_data()

def create_labels_for_case(case_dir: Path, timestamps: List[int], all_services: List[str]) -> List[torch.Tensor]:
    """为单个case创建标签"""
    env_path = case_dir / 'env.json'
    inj_path = case_dir / 'injection.json'
    
    if not env_path.exists() or not inj_path.exists():
        # 如果没有注入信息，返回全零标签
        print("no label:", case_dir)
        return [torch.zeros(len(all_services)) for _ in timestamps]
    
    with open(env_path) as f:
        env = json.load(f)
    with open(inj_path) as f:
        inj = json.load(f)
    
    # 提取故障服务
    injection_name = extract_injection_name(inj['injection_name'])
    
    
    fault_service_idx = None
    for i, service in enumerate(all_services):
        if injection_name in service or service in injection_name:
            fault_service_idx = i
            break
    # 创建标签
    labels = []
    for timestamp in timestamps:
        label = torch.zeros(len(all_services))
        if fault_service_idx is not None:
            # 检查时间是否在故障时间范围内
            ab_start=pd.to_datetime(int(env['ABNORMAL_START']), unit='s', utc=True)-pd.Timedelta(hours=8)
            ab_end=pd.to_datetime(int(env['ABNORMAL_END']), unit='s', utc=True)-pd.Timedelta(hours=8)
            ab_start=int(ab_start.timestamp())
            ab_end=int(ab_end.timestamp())
            window_start = timestamp * SAMPLING_SIZE
            window_end = (timestamp + 1) * SAMPLING_SIZE
            
            # 如果窗口与故障时间有重叠，则标记为故障
            if not (window_end <= ab_start or window_start >= ab_end):
                label[fault_service_idx] = 1.0
        labels.append(label)
    
    return labels


def build_groundtruth(case_dirs: List[Path], output_csv: str):
    records = []
    for case_dir in case_dirs:
        env_path = case_dir / 'env.json'
        inj_path = case_dir / 'injection.json'
        if not env_path.exists() or not inj_path.exists():
            continue
        with open(env_path) as f:
            env = json.load(f)
        with open(inj_path) as f:
            inj = json.load(f)
        inj_name = extract_injection_name(inj['injection_name'])
        failure_type = inj['fault_type']
        ab_start=pd.to_datetime(int(env['ABNORMAL_START']), unit='s', utc=True)-pd.Timedelta(hours=8)
        ab_end=pd.to_datetime(int(env['ABNORMAL_END']), unit='s', utc=True)-pd.Timedelta(hours=8)
        ab_start=int(ab_start.timestamp())
        ab_end=int(ab_end.timestamp())

        aligned_start = ab_start - (ab_start % SAMPLING_SIZE)
        for t in range(aligned_start, ab_end, SAMPLING_SIZE):
            # 直接存储为int秒
            records.append({
                'timestamp': t,
                'injection_name': inj_name,
                'failure_type': failure_type
            })
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return records

def create_dataset_for_training(case_data_list: List[Tuple[List[int], List[torch.Tensor], List[List[Dict]]]], 
                               all_services: List[str],
                               window_size: int = WINDOW_SIZE, max_gap: int = 15) -> List[Tuple[torch.Tensor, dgl.DGLGraph]]:
    """对所有case的数据做滑动窗口，合并为训练集"""
    all_samples = []
    for timestamps, features, edges in case_data_list:
        pack = DatasetPack(timestamps, features, edges)
        all_samples.extend(pack.create_sliding_windows(window_size=window_size, max_gap=max_gap, all_services=all_services))
    return all_samples

def main():
    output_dir = 'data/RCABENCH'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    
    # 获取数据包路径
    data_packs = create_dataset(
        data_root='/mnt/jfs/rcabench-platform-v2/data/rcabench_filtered',
        output_dir='data',
        max_cases=30,
        ds='rcabench_filtered'
    )
    print(f"找到 {len(data_packs)} 个数据包")
    
    # 收集所有服务和指标
    all_services, all_metrics = collect_all_services_and_metrics(data_packs, 'abnormal_metrics.parquet')
    print(f"找到 {len(all_services)} 个服务: {all_services}")
    print(f"找到 {len(all_metrics)} 个指标: {all_metrics}")
    
    # 保存全局service_to_idx和metric_to_idx
    service_to_idx = {service: i for i, service in enumerate(all_services)}
    metric_to_idx = {metric: i for i, metric in enumerate(all_metrics)}
    with open(os.path.join(output_dir, 'service_to_idx.pkl'), 'wb') as f:
        pickle.dump(service_to_idx, f)
    with open(os.path.join(output_dir, 'metric_to_idx.pkl'), 'wb') as f:
        pickle.dump(metric_to_idx, f)

    # 处理异常数据
    abnormal_case_data = []
    abnormal_labels_by_case = []
    
    logger.info("处理异常数据...")
    for case_dir in tqdm(data_packs, desc='Processing abnormal cases'):
        try:
            timestamps, features, edges = process_case(
                case_dir, 'abnormal_metrics.parquet', 'abnormal_traces.parquet', 
                'abnormal_logs.parquet', all_services, all_metrics
            )
            if timestamps:
                abnormal_case_data.append((timestamps, features, edges))
                labels = create_labels_for_case(case_dir, timestamps, all_services)
                abnormal_labels_by_case.append(labels)
        except Exception as e:
            logger.warning(f"处理case {case_dir} 时出错: {e}")
            continue
    # 标准化特征
    logger.info("标准化特征...")
    all_features = []
    for _, features, _ in abnormal_case_data:
        all_features.extend(features)
    
    if all_features:
        normalized_features = minmax_normalize_features(all_features)
        
        # 替换原始特征
        idx = 0
        for i, (timestamps, _, edges) in enumerate(abnormal_case_data):
            num_windows = len(timestamps)
            abnormal_case_data[i] = (
                timestamps, 
                normalized_features[idx:idx+num_windows], 
                edges
            )
            idx += num_windows

    # 创建滑动窗口样本
    logger.info("创建滑动窗口样本...")
    abnormal_samples = create_dataset_for_training(abnormal_case_data, all_services, window_size=WINDOW_SIZE, max_gap=15)
    
    # 创建对应的标签
    abnormal_sample_labels = []
    sample_idx = 0
    for case_labels in abnormal_labels_by_case:
        if len(case_labels) >= WINDOW_SIZE:
            for i in range(len(case_labels) - WINDOW_SIZE + 1):
                # 使用最后一个时间的标签作为样本标签
                abnormal_sample_labels.append(case_labels[i + WINDOW_SIZE - 1])
                sample_idx += 1

    # 保存数据
    logger.info(f"保存 {len(abnormal_samples)} 个异常样本...")
    with open(os.path.join(output_dir, 'samples', 'abnormal_samples_latency.pkl'), 'wb') as f:
        pickle.dump(abnormal_samples, f)
    
    with open(os.path.join(output_dir, 'samples', 'abnormal_labels_latency.pkl'), 'wb') as f:
        pickle.dump(abnormal_sample_labels, f)
    
    logger.info("数据预处理完成！")
    
    # 输出统计信息
    if abnormal_samples:
        print(len(abnormal_samples), "个异常样本")
        sample_features, sample_graph = abnormal_samples[0]
        print(f"样本特征形状: {sample_features.shape}")
        print(f"图节点数: {sample_graph.num_nodes()}")
        print(f"图边数: {sample_graph.num_edges()}")
        print(f"标签形状: {abnormal_sample_labels[0].shape if abnormal_sample_labels else 'N/A'}")
        print(abnormal_sample_labels[0])


if __name__ == '__main__':
    main()
    # path= Path(__file__).parent / 'data/RCABENCH/samples/abnormal_samples.pkl'
    # with open(path, 'rb') as f:
    #     abnormal_samples = pickle.load(f)
    # print(abnormal_samples[1][0].shape)
    # print(abnormal_samples[1])
    # for i in range(10):
    #     print(abnormal_samples[i][0][1,1,-6:-1])
