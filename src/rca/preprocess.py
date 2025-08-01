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

import statistics  # For trace latency std calculation
import typer

from enum import Enum, auto  # Import Enum base class and auto value generator


class Dataset(Enum):  # Define an Enum class named Dataset, inheriting from Enum
    RCABENCH_database = auto()
    RCABENCH_filtered = auto()
    RCABENCH_r1 = auto()
    RCABENCH_r2 = auto()


# Configuration parameters
SAMPLING_SIZE = 5  # Data smoothing granularity: 5 seconds
WINDOW_SIZE = 20  # Sliding window size: 10 sampling points
SLIDING_STEP = 5  # Sliding step size
    

from typing import Union

def create_dataset(
    data_root: str,
    output_dir: str,
    max_cases: Optional[int] = None,
    ds: str = "RCABENCH_r1",
) -> Union[List[Path], Tuple[List[Path], List[Path]]]:
    if ds == "rcabench":
        dataset = Dataset.RCABENCH_database
    elif ds == "rcabench_filtered":
        dataset = Dataset.RCABENCH_filtered
    elif ds == "RCABENCH_r1":
        dataset = Dataset.RCABENCH_r1
    elif ds == "RCABENCH_r2":
        dataset = Dataset.RCABENCH_r2

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
        # case_names = os.listdir(data_root)
        data_packs = [Path(data_root) / name / "converted" for name in case_names]

        data_packs = data_packs[:max_cases]

    elif dataset == Dataset.RCABENCH_filtered:
        logger.info(f"Using filtered dataset from {data_root}")
        service_path_name = []

        # Traverse directory
        for root, dirs, files in os.walk(data_root):
            for dir_name in dirs:
                # if any(sample in dir_name for sample in service_select):
                if "ts" in dir_name:
                    service_path_name.append(dir_name)

        data_packs = [Path(data_root) / name for name in service_path_name]

        data_packs = data_packs[:max_cases]

    elif dataset == Dataset.RCABENCH_r1 or dataset == Dataset.RCABENCH_r2:
        logger.info(f"Using {ds} dataset from {data_root}")
        # Use RCABENCH_r1 or RCABENCH_r2 dataset
        if dataset == Dataset.RCABENCH_r1:
            train_dir = Path(data_root) / "__dev__rcabench_train_r1"
            test_dir = Path(data_root) / "__dev__rcabench_test_r1"
        else:
            train_dir = Path(data_root) / "__dev__rcabench_train_r2"
            test_dir = Path(data_root) / "__dev__rcabench_test_r2"

        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError("Train or test directory does not exist")

        train_cases = os.listdir(train_dir)
        test_cases = os.listdir(test_dir)

        train_data_packs = [train_dir / case for case in train_cases]
        test_data_packs = [test_dir / case for case in test_cases]

        if max_cases is not None:
            train_data_packs = train_data_packs[:max_cases]
            test_data_packs = test_data_packs[:max_cases]
        
        return train_data_packs, test_data_packs

    return data_packs


class DataPreprocessor:
    def __init__(
        self,
        metrics_path: str,
        traces_path: str,
        logs_path: str,
        all_services: List[str],
        all_metrics: List[str],
        sampling_size: int = SAMPLING_SIZE,
    ):
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

        # Store processed data
        self.features_by_window: Optional[Dict[int, torch.Tensor]] = None
        self.trace_edges_by_window: Optional[Dict[int, List[Dict]]] = (
            None  # Store raw edge info
        )
        self.time_windows: Optional[List[int]] = None

    @staticmethod
    def extract_service_name(row: pd.Series) -> Optional[str]:
        if row["service_name"] != "":
            return row["service_name"]
        elif "attr.k8s.container.name" in row and row["attr.k8s.container.name"] != "":
            return row["attr.k8s.container.name"]
        elif "attr.k8s.service.name" in row and row["attr.k8s.service.name"] != "":
            return row["attr.k8s.service.name"]
        return None

    def load_data(self) -> None:
        self.metrics_df = pd.read_parquet(self.metrics_path)
        self.metrics_df["value"] = self.metrics_df["value"].fillna(0)
        self.traces_df = pd.read_parquet(self.traces_path)
        self.logs_df = pd.read_parquet(self.logs_path)

    def preprocess_data(self) -> None:
        """Unified processing of metrics, traces, and logs data, extract features and build graph"""
        if any(df is None for df in [self.metrics_df, self.traces_df, self.logs_df]):
            raise ValueError("Data not loaded, please call load_data() first")

        # 1. Process metrics data
        metrics_features = self._process_metrics()

        # 2. Process traces data, only extract features and edge info, do not build graph
        trace_features, trace_edges = self._process_traces()

        # 3. Process logs data
        log_features = self._process_logs()

        # 4. Combine all features
        self._combine_features(metrics_features, trace_features, log_features)

        # 5. Save edge info for later graph construction
        self.trace_edges_by_window = trace_edges

        if self.time_windows:
            logger.info(
                f"Processing completed, {len(self.time_windows)} time windows in total"
            )

    def _process_metrics(self) -> Dict[int, torch.Tensor]:
        """Process metrics data"""
        if self.metrics_df is None:
            return {}

        df = self.metrics_df.copy()
        df["extracted_service"] = df.apply(
            DataPreprocessor.extract_service_name, axis=1
        )  # type: ignore
        df = df.dropna(subset=["extracted_service"]).copy()
        df.loc[:, "timestamp"] = df["time"].astype("int64") // 1_000_000_000
        df.loc[:, "time_window"] = df["timestamp"] // self.sampling_size

        # Aggregate by time window, service, metric
        grouped = (
            df.groupby(["time_window", "extracted_service", "metric"])["value"]
            .mean()
            .reset_index()
        )

        # Build feature matrix for each time window, using global metrics set
        features_by_window = {}
        time_windows = sorted(grouped["time_window"].unique())

        for window in time_windows:
            window_data = grouped[grouped["time_window"] == window]
            # Use the number of global metrics to create feature matrix
            feature_matrix = torch.zeros((len(self.services), len(self.metrics)))

            for _, row in window_data.iterrows():
                service_idx = self.service_to_idx.get(row["extracted_service"])
                metric_idx = self.metric_to_idx.get(row["metric"])
                if service_idx is not None and metric_idx is not None:
                    feature_matrix[service_idx, metric_idx] = row["value"]

            features_by_window[window] = feature_matrix

        return features_by_window

    def _process_traces(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[Dict]]]:
        """Process traces data, return features and edge info, and extract trace latency features"""
        if self.traces_df is None:
            return {}, {}
        df = self.traces_df.copy()
        df["extracted_service"] = df.apply(
            DataPreprocessor.extract_service_name, axis=1
        )  # type: ignore
        df = df.dropna(subset=["extracted_service"]).copy()
        df.loc[:, "timestamp"] = df["time"].astype("int64") // 1_000_000_000
        df.loc[:, "time_window"] = df["timestamp"] // self.sampling_size
        # Build service call relationships
        df["source_service"] = df["extracted_service"]
        df["destination_service"] = None
        span_service_map = df.set_index("span_id")["extracted_service"].to_dict()

        def find_destination(row) -> Optional[str]:
            parent_span_id = row["parent_span_id"]
            if parent_span_id in span_service_map:
                return span_service_map[parent_span_id]
            return None

        df["destination_service"] = df.apply(find_destination, axis=1)  # type: ignore
        df = df.dropna(subset=["source_service", "destination_service"]).copy()
        df["is_error"] = df["attr.status_code"].isin(["Error"])
        trace_features = {}
        trace_edges = {}
        time_windows = sorted(df["time_window"].unique())
        for window in time_windows:
            window_data = df[df["time_window"] == window]
            # 1. Extract trace features (duration, request_count, latency_p90, latency_std)
            trace_stats = (
                window_data.groupby("extracted_service")
                .agg({"duration": "mean", "trace_id": "count"})
                .reset_index()
            )
            trace_stats.columns = ["service", "avg_duration", "request_count"]
            feature_matrix = torch.zeros(
                (len(self.services), 2)
            )  # duration, count, latency_p90, latency_std
            for idx, row in trace_stats.iterrows():
                service_idx = self.service_to_idx.get(row["service"])
                if service_idx is not None:
                    feature_matrix[service_idx, 0] = row["avg_duration"]
                    feature_matrix[service_idx, 1] = row["request_count"]

            # 2. Collect edge info (do not build graph immediately)
            edge_stats = (
                window_data.groupby(["source_service", "destination_service"])
                .agg(
                    {
                        "trace_id": "count",  # Call frequency
                        "is_error": "mean",  # Error rate
                    }
                )
                .reset_index()
            )
            edge_stats.columns = [
                "src_service",
                "dst_service",
                "call_frequency",
                "error_rate",
            ]

            edge_list = []
            for _, row in edge_stats.iterrows():
                src_idx = self.service_to_idx.get(row["src_service"])
                dst_idx = self.service_to_idx.get(row["dst_service"])

                if src_idx is not None and dst_idx is not None and src_idx != dst_idx:
                    edge_list.append(
                        {
                            "src": src_idx,
                            "dst": dst_idx,
                            "call_frequency": row["call_frequency"],
                            "error_rate": row["error_rate"],
                        }
                    )

            trace_edges[window] = edge_list

        return trace_features, trace_edges

    def _process_logs(self) -> Dict[int, torch.Tensor]:
        """Process logs data"""
        if self.logs_df is None:
            return {}

        df = self.logs_df.copy()
        df["extracted_service"] = df.apply(
            DataPreprocessor.extract_service_name, axis=1
        )  # type: ignore
        df = df.dropna(subset=["extracted_service"]).copy()
        df.loc[:, "timestamp"] = df["time"].astype("int64") // 1_000_000_000
        df.loc[:, "time_window"] = df["timestamp"] // self.sampling_size

        # Count the number of logs at each level
        log_levels = ["INFO", "ERROR", "WARN", "DEBUG"]
        log_counts = (
            df.groupby(["time_window", "extracted_service", "level"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        # Ensure all levels exist
        for level in log_levels:
            if level not in log_counts.columns:
                log_counts[level] = 0

        # Build log features for each time window
        features_by_window = {}
        time_windows = sorted(log_counts["time_window"].unique())

        for window in time_windows:
            window_data = log_counts[log_counts["time_window"] == window]
            feature_matrix = torch.zeros((len(self.services), len(log_levels)))

            for _, row in window_data.iterrows():
                service_idx = self.service_to_idx.get(row["extracted_service"])
                if service_idx is not None:
                    for i, level in enumerate(log_levels):
                        feature_matrix[service_idx, i] = row[level]

            features_by_window[window] = feature_matrix

        return features_by_window

    def _combine_features(
        self,
        metrics_features: Dict[int, torch.Tensor],
        trace_features: Dict[int, torch.Tensor],
        log_features: Dict[int, torch.Tensor],
    ) -> None:
        """Combine all features"""
        # Get all time windows
        all_windows = (
            set(metrics_features.keys())
            | set(trace_features.keys())
            | set(log_features.keys())
        )
        self.time_windows = sorted(all_windows)

        combined_features = {}
        for window in self.time_windows:
            feature_list = []

            # Add metrics features
            if window in metrics_features:
                feature_list.append(metrics_features[window])
            else:
                # If there is no metrics data for this window, fill with zeros
                feature_list.append(
                    torch.zeros((len(self.services), len(self.metrics)))
                )

            # Add trace features
            if window in trace_features:
                feature_list.append(trace_features[window])
            else:
                feature_list.append(torch.zeros((len(self.services), 2)))

            # Add log features
            if window in log_features:
                feature_list.append(log_features[window])
            else:
                feature_list.append(torch.zeros((len(self.services), 4)))

            # Concatenate all features
            combined_features[window] = torch.cat(feature_list, dim=1)

        self.features_by_window = combined_features

    def combine_data(self) -> Tuple[List[int], List[torch.Tensor], List[List[Dict]]]:
        """Return processed data for sliding window sampling"""
        if (
            self.features_by_window is None
            or self.trace_edges_by_window is None
            or self.time_windows is None
        ):
            raise ValueError("Data not processed, please call preprocess_data() first")

        timestamps = []
        features = []
        edges = []

        for window in self.time_windows:
            timestamps.append(window)
            features.append(self.features_by_window[window])
            edges.append(self.trace_edges_by_window.get(window, []))

        return timestamps, features, edges


class DatasetPack:
    """Data pack for each case, including sliding window sample extraction"""

    def __init__(
        self,
        timestamps: List[int],
        features: List[torch.Tensor],
        edges: List[List[Dict]],
    ):
        self.timestamps = timestamps
        self.features = features
        self.edges = edges  # Edge info list for each time window

    def create_sliding_windows(
        self,
        window_size: int = WINDOW_SIZE,
        max_gap: int = 15,
        all_services: Optional[List[str]] = None,
    ) -> List[Tuple[torch.Tensor, dgl.DGLGraph]]:
        """Create sliding window samples, build graph based on the entire sliding window"""
        if len(self.features) < window_size:
            return []

        if all_services is None:
            all_services = [f"service_{i}" for i in range(self.features[0].shape[0])]

        samples = []
        for i in range(0, len(self.features) - window_size + 1, SLIDING_STEP):
            # Check if time interval is reasonable
            time_diffs = [
                abs(self.timestamps[i + j + 1] - self.timestamps[i + j])
                for j in range(window_size - 1)
            ]

            if all(diff <= max_gap for diff in time_diffs):
                # Feature sequence (window_size, num_nodes, num_features)
                feature_sequence = torch.stack(self.features[i : i + window_size])

                # Build graph based on the entire sliding window
                window_graph = self._build_graph_from_window(
                    self.edges[i : i + window_size], feature_sequence, len(all_services)
                )

                samples.append((feature_sequence, window_graph))

        return samples

    def _build_graph_from_window(
        self,
        window_edges: List[List[Dict]],
        window_features: torch.Tensor,
        num_nodes: int,
    ) -> dgl.DGLGraph:
        """Build a comprehensive graph based on all edge info and features in the sliding window"""

        # Count edge connection info in the entire window
        edge_stats = defaultdict(lambda: {"call_frequencies": [], "error_rates": []})

        # Collect all edge info in the window
        for edge_list in window_edges:
            for edge in edge_list:
                key = (edge["src"], edge["dst"])
                edge_stats[key]["call_frequencies"].append(edge["call_frequency"])
                edge_stats[key]["error_rates"].append(edge["error_rate"])

        # Build graph
        graph = dgl.graph(([], []), num_nodes=num_nodes)

        if edge_stats:
            src_list = []
            dst_list = []
            weight_list = []

            # Collect all edge weights for normalization
            all_call_freqs = []
            all_error_rates = []

            for (src, dst), stats in edge_stats.items():
                avg_call_freq = np.mean(stats["call_frequencies"])
                avg_error_rate = np.mean(stats["error_rates"])
                all_call_freqs.append(avg_call_freq)
                all_error_rates.append(avg_error_rate)

            # Normalization
            if all_call_freqs:
                call_freq_array = np.array(all_call_freqs)
                error_rate_array = np.array(all_error_rates)

                # Min-Max normalization
                def minmax_normalize_np(x):
                    min_val = x.min()
                    max_val = x.max()
                    if max_val - min_val < 1e-8:
                        return np.zeros_like(x)
                    return (x - min_val) / (max_val - min_val)

                norm_call_freqs = minmax_normalize_np(call_freq_array)
                norm_error_rates = minmax_normalize_np(error_rate_array)

                # Build edges
                for idx, ((src, dst), _) in enumerate(edge_stats.items()):
                    src_list.append(src)
                    dst_list.append(dst)
                    # Weighted combination
                    weight = 0.5 * norm_call_freqs[idx] + 0.5 * norm_error_rates[idx]
                    weight_list.append(weight)

                graph.add_edges(src_list, dst_list)
                graph.edata["weight"] = torch.tensor(weight_list, dtype=torch.float32)

        return graph


def extract_injection_name(injection_name: str) -> str:
    parts = injection_name.split("-")
    for i, p in enumerate(parts):
        if p == "service" and i > 0:
            return "-".join(parts[: i + 1])
    return injection_name


def minmax_normalize_features(feature_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """Min-Max normalization for feature list"""
    if not feature_list:
        return []

    # Concatenate all features
    all_features = torch.stack(feature_list)  # (num_samples, num_nodes, num_features)

    # Calculate min and max for each feature dimension
    min_vals = all_features.min(dim=0, keepdim=True)[0].min(dim=0, keepdim=True)[
        0
    ]  # (1, 1, num_features)
    max_vals = all_features.max(dim=0, keepdim=True)[0].max(dim=0, keepdim=True)[
        0
    ]  # (1, 1, num_features)

    # Avoid denominator being 0
    range_vals = max_vals - min_vals
    range_vals[range_vals < 1e-8] = 1.0

    # Normalization
    normalized = (all_features - min_vals) / range_vals

    return [normalized[i] for i in range(normalized.shape[0])]


def collect_all_services_and_metrics(
    case_dirs: List[Path], metric_file_name: str
) -> Tuple[List[str], List[str]]:
    """Collect all service names and metric names"""
    all_services = set()
    all_metrics = set()

    for case_dir in case_dirs:
        metric_path = case_dir / metric_file_name
        if not metric_path.exists():
            continue
        df = pd.read_parquet(metric_path)
        df["extracted_service"] = df.apply(
            DataPreprocessor.extract_service_name, axis=1
        )  # type: ignore

        # Collect services
        services = df["extracted_service"].dropna().unique()
        all_services.update(services)

        # Collect metrics
        metrics = df["metric"].unique()
        all_metrics.update(metrics)

        typer.echo(f"Case {case_dir.name}: {len(services)} services, {len(metrics)} metrics")

    return sorted(all_services), sorted(all_metrics)


def process_case(
    case_dir: Path,
    metric_file: str,
    trace_file: str,
    log_file: str,
    all_services: List[str],
    all_metrics: List[str],
    sampling_size: int = SAMPLING_SIZE,
) -> Tuple[List[int], List[torch.Tensor], List[List[Dict]]]:
    """Process a single case, return timestamps, features, and edge info"""
    processor = DataPreprocessor(
        metrics_path=os.path.join(case_dir, metric_file),
        traces_path=os.path.join(case_dir, trace_file),
        logs_path=os.path.join(case_dir, log_file),
        all_services=all_services,
        all_metrics=all_metrics,
        sampling_size=sampling_size,
    )
    processor.load_data()
    processor.preprocess_data()
    return processor.combine_data()


def create_labels_for_case(
    case_dir: Path, timestamps: List[int], all_services: List[str]
) -> List[torch.Tensor]:
    """Create labels for a single case"""
    env_path = case_dir / "env.json"
    inj_path = case_dir / "injection.json"

    if not env_path.exists() or not inj_path.exists():
        # If no injection info, return all-zero labels
        typer.echo(f"no label: {case_dir}")
        return [torch.zeros(len(all_services)) for _ in timestamps]

    with open(env_path) as f:
        env = json.load(f)
    with open(inj_path) as f:
        inj = json.load(f)

    # Extract fault service
    injection_name = extract_injection_name(inj["injection_name"])

    fault_service_idx = None
    for i, service in enumerate(all_services):
        if injection_name in service or service in injection_name:
            fault_service_idx = i
            break
    # Create labels
    labels = []
    for timestamp in timestamps:
        label = torch.zeros(len(all_services))
        if fault_service_idx is not None:
            # Check if the time is within the fault time range
            ab_start = pd.to_datetime(
                int(env["ABNORMAL_START"]), unit="s", utc=True
            )
            ab_end = pd.to_datetime(
                int(env["ABNORMAL_END"]), unit="s", utc=True
            )
            ab_start = int(ab_start.timestamp())
            ab_end = int(ab_end.timestamp())
            window_start = timestamp * SAMPLING_SIZE
            window_end = (timestamp + 1) * SAMPLING_SIZE

            # If the window overlaps with the fault time, mark as fault
            if not (window_end <= ab_start or window_start >= ab_end):
                label[fault_service_idx] = 1.0
        labels.append(label)

    return labels


def build_groundtruth(case_dirs: List[Path], output_csv: str):
    records = []
    for case_dir in case_dirs:
        env_path = case_dir / "env.json"
        inj_path = case_dir / "injection.json"
        if not env_path.exists() or not inj_path.exists():
            continue
        with open(env_path) as f:
            env = json.load(f)
        with open(inj_path) as f:
            inj = json.load(f)
        inj_name = extract_injection_name(inj["injection_name"])
        failure_type = inj["fault_type"]
        ab_start = pd.to_datetime(
            int(env["ABNORMAL_START"]), unit="s", utc=True
        ) - pd.Timedelta(hours=8)
        ab_end = pd.to_datetime(
            int(env["ABNORMAL_END"]), unit="s", utc=True
        ) - pd.Timedelta(hours=8)
        ab_start = int(ab_start.timestamp())
        ab_end = int(ab_end.timestamp())

        aligned_start = ab_start - (ab_start % SAMPLING_SIZE)
        for t in range(aligned_start, ab_end, SAMPLING_SIZE):
            # Store as int seconds directly
            records.append(
                {
                    "timestamp": t,
                    "injection_name": inj_name,
                    "failure_type": failure_type,
                }
            )
    pd.DataFrame(records).to_csv(output_csv, index=False)
    return records


def create_dataset_for_training(
    case_data_list: List[Tuple[List[int], List[torch.Tensor], List[List[Dict]]]],
    all_services: List[str],
    window_size: int = WINDOW_SIZE,
    max_gap: int = 15,
) -> List[Tuple[torch.Tensor, dgl.DGLGraph]]:
    """Apply sliding window to all case data and merge into training set"""
    all_samples = []
    for timestamps, features, edges in case_data_list:
        pack = DatasetPack(timestamps, features, edges)
        all_samples.extend(
            pack.create_sliding_windows(
                window_size=window_size, max_gap=max_gap, all_services=all_services
            )
        )
    return all_samples


def main():
    output_dir = "data/RCABENCH"
    
    # Use RCABENCH_r1 dataset which returns train/test splits
    result = run_preprocessing(
        data_root="/mnt/jfs/rcabench-platform-v2/data",  # Adjust path as needed
        output_dir=output_dir,
        max_cases=30,
        ds="RCABENCH_r1",  # Use the train/test split dataset
    )
    
    typer.echo("Preprocessing completed!")
    typer.echo(f"Train samples: {result['num_train_samples']}")
    typer.echo(f"Test samples: {result['num_test_samples']}")
    typer.echo(f"Total services: {result['num_services']}")
    typer.echo(f"Total metrics: {result['num_metrics']}")


def run_preprocessing(
    data_root: str,
    output_dir: str,
    max_cases: Optional[int] = None,
    ds: str = "rcabench",
):
    """Main function to run data preprocessing for both train and test sets"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

    # Get data pack paths - now returns train and test separately
    dataset_result = create_dataset(
        data_root=data_root, output_dir="data", max_cases=max_cases, ds=ds
    )
    
    # Handle different return types based on dataset
    if isinstance(dataset_result, tuple):
        train_data_packs, test_data_packs = dataset_result
        typer.echo(f"Found {len(train_data_packs)} train data packs")
        typer.echo(f"Found {len(test_data_packs)} test data packs")
        all_data_packs = train_data_packs + test_data_packs
    else:
        # For backward compatibility with single dataset
        all_data_packs = dataset_result
        train_data_packs = all_data_packs
        test_data_packs = []
        typer.echo(f"Found {len(all_data_packs)} data packs (single dataset)")

    # Collect all services and metrics from both train and test
    all_services, all_metrics = collect_all_services_and_metrics(
        all_data_packs, "abnormal_metrics.parquet"
    )
    typer.echo(f"Found {len(all_services)} services: {all_services}")
    typer.echo(f"Found {len(all_metrics)} metrics: {all_metrics}")

    # Save global service_to_idx and metric_to_idx (unified for train and test)
    service_to_idx = {service: i for i, service in enumerate(all_services)}
    metric_to_idx = {metric: i for i, metric in enumerate(all_metrics)}
    with open(os.path.join(output_dir, "service_to_idx_r2.pkl"), "wb") as f:
        pickle.dump(service_to_idx, f)
    with open(os.path.join(output_dir, "metric_to_idx_r2.pkl"), "wb") as f:
        pickle.dump(metric_to_idx, f)

    def process_data_packs(data_packs, data_type="train"):
        """Process a list of data packs and return samples and labels"""
        case_data = []
        labels_by_case = []

        logger.info(f"Processing {data_type} data...")
        for case_dir in tqdm(data_packs, desc=f"Processing {data_type} cases"):
            try:
                timestamps, features, edges = process_case(
                    case_dir,
                    "abnormal_metrics.parquet",
                    "abnormal_traces.parquet",
                    "abnormal_logs.parquet",
                    all_services,
                    all_metrics,
                )
                if timestamps:
                    case_data.append((timestamps, features, edges))
                    labels = create_labels_for_case(case_dir, timestamps, all_services)
                    labels_by_case.append(labels)
            except Exception as e:
                logger.warning(f"Error processing case {case_dir}: {e}")
                continue

        # Normalize features
        logger.info(f"Normalizing {data_type} features...")
        all_features = []
        for _, features, _ in case_data:
            all_features.extend(features)

        if all_features:
            normalized_features = minmax_normalize_features(all_features)

            # Replace original features
            idx = 0
            for i, (timestamps, _, edges) in enumerate(case_data):
                num_windows = len(timestamps)
                case_data[i] = (
                    timestamps,
                    normalized_features[idx : idx + num_windows],
                    edges,
                )
                idx += num_windows

        # Create sliding window samples
        logger.info(f"Creating {data_type} sliding window samples...")
        samples = create_dataset_for_training(
            case_data, all_services, window_size=WINDOW_SIZE, max_gap=15
        )

        # Create corresponding labels
        sample_labels = []
        for case_labels in labels_by_case:
            if len(case_labels) >= WINDOW_SIZE:
                for i in range(0, len(case_labels) - WINDOW_SIZE + 1, SLIDING_STEP):
                    # Use the label of the last time as the sample label
                    sample_labels.append(case_labels[i + WINDOW_SIZE - 1])

        return samples, sample_labels

    # Process train data
    if train_data_packs:
        train_samples, train_labels = process_data_packs(train_data_packs, "train")
        
        # Save train data
        logger.info(f"Saving {len(train_samples)} train samples...")
        with open(os.path.join(output_dir, "samples", "train_samples_r2.pkl"), "wb") as f:
            pickle.dump(train_samples, f)
        with open(os.path.join(output_dir, "samples", "train_labels_r2.pkl"), "wb") as f:
            pickle.dump(train_labels, f)

        # Output train statistics
        if train_samples:
            typer.echo(f"Train: {len(train_samples)} samples")
            sample_features, sample_graph = train_samples[0]
            typer.echo(f"Train sample feature shape: {sample_features.shape}")
            typer.echo(f"Train graph node count: {sample_graph.num_nodes()}")
            typer.echo(f"Train graph edge count: {sample_graph.num_edges()}")
            typer.echo(f"Train label shape: {train_labels[0].shape if train_labels else 'N/A'}")
    else:
        train_samples, train_labels = [], []

    # Process test data
    if test_data_packs:
        test_samples, test_labels = process_data_packs(test_data_packs, "test")
        
        # Save test data
        logger.info(f"Saving {len(test_samples)} test samples...")
        with open(os.path.join(output_dir, "samples", "test_samples_r2.pkl"), "wb") as f:
            pickle.dump(test_samples, f)
        with open(os.path.join(output_dir, "samples", "test_labels_r2.pkl"), "wb") as f:
            pickle.dump(test_labels, f)

        # Output test statistics
        if test_samples:
            typer.echo(f"Test: {len(test_samples)} samples")
            sample_features, sample_graph = test_samples[0]
            typer.echo(f"Test sample feature shape: {sample_features.shape}")
            typer.echo(f"Test graph node count: {sample_graph.num_nodes()}")
            typer.echo(f"Test graph edge count: {sample_graph.num_edges()}")
            typer.echo(f"Test label shape: {test_labels[0].shape if test_labels else 'N/A'}")
    else:
        test_samples, test_labels = [], []

    logger.info("Data preprocessing completed!")

    return {
        "num_train_samples": len(train_samples),
        "num_test_samples": len(test_samples),
        "num_services": len(all_services),
        "num_metrics": len(all_metrics),
        "train_samples_path": os.path.join(output_dir, "samples", "train_samples.pkl") if train_data_packs else None,
        "train_labels_path": os.path.join(output_dir, "samples", "train_labels.pkl") if train_data_packs else None,
        "test_samples_path": os.path.join(output_dir, "samples", "test_samples.pkl") if test_data_packs else None,
        "test_labels_path": os.path.join(output_dir, "samples", "test_labels.pkl") if test_data_packs else None,
    }


def run_single_pack_preprocessing(
        datapack_path: Path,
        output_dir: str,
        metric_file: str = "abnormal_metrics.parquet",
        trace_file: str = "abnormal_traces.parquet",
        log_file: str = "abnormal_logs.parquet",
        sampling_size: int = SAMPLING_SIZE,
        window_size: int = WINDOW_SIZE,
        max_gap: int = 15,
) -> dict:

        output_dir = str(output_dir)
        # load all_services and all_metrics
        with open(os.path.join(output_dir, "service_to_idx.pkl"), "rb") as f:
            service_to_idx = pickle.load(f)
        with open(os.path.join(output_dir, "metric_to_idx.pkl"), "rb") as f:
            metric_to_idx = pickle.load(f)
        all_services = list(service_to_idx.keys())
        all_metrics = list(metric_to_idx.keys())

        # process single data pack
        timestamps, features, edges = process_case(
            datapack_path,
            metric_file,
            trace_file,
            log_file,
            all_services,
            all_metrics,
            sampling_size=sampling_size,
        )

        # normalize features
        if features:
            normalized_features = minmax_normalize_features(features)
        else:
            normalized_features = []

        # sliding window sampling
        pack = DatasetPack(timestamps, normalized_features, edges)
        samples = pack.create_sliding_windows(
            window_size=window_size, max_gap=max_gap, all_services=all_services
        )

        # create labels
        labels = create_labels_for_case(datapack_path, timestamps, all_services)
        sample_labels = []
        if len(labels) >= window_size:
            for i in range(len(labels) - window_size + 1):
                sample_labels.append(labels[i + window_size - 1])

        # save samples and labels
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
        with open(os.path.join(output_dir, "samples", "single_pack_samples.pkl"), "wb") as f:
            pickle.dump(samples, f)
        with open(os.path.join(output_dir, "samples", "single_pack_labels.pkl"), "wb") as f:
            pickle.dump(sample_labels, f)

        typer.echo(f"单个 data_pack 处理完成，样本数: {len(samples)}，标签数: {len(sample_labels)}")
        if samples:
            typer.echo(f"Sample feature shape: {samples[0][0].shape}")
            typer.echo(f"Graph: {samples[0][1]}")
        return {"samples": samples, "sample_labels": sample_labels}


if __name__ == "__main__":
    # main()
    data_pack=Path("/mnt/jfs/rcabench_dataset/ts5-ts-ui-dashboard-request-replace-method-fjhvwr/converted")
    x=run_single_pack_preprocessing(data_pack, output_dir="data/RCABENCH")
    # path= Path(__file__).parent / 'data/RCABENCH/samples/abnormal_samples.pkl'
    # with open(path, 'rb') as f:
    #     abnormal_samples = pickle.load(f)
    # print(abnormal_samples[1][0].shape)
    # print(abnormal_samples[1])
    # for i in range(10):
    #     print(abnormal_samples[i][0][1,1,-6:-1])
