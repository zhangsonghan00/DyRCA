from rcabench_platform.v2.cli.main import main
from rcabench_platform.v2.algorithms.spec import (
    global_algorithm_registry,
    Algorithm,
    AlgorithmArgs,
    AlgorithmAnswer,
)
import os
from client import run_inference

class PadiRCA(Algorithm):
    def needs_cpu_count(self) -> int | None:
        return 4

    def __call__(self, args: AlgorithmArgs) -> list[AlgorithmAnswer]:
        model_path = os.environ["MODEL_PATH"]
        results = run_inference(
            datapack_path=args.input_folder,
            model_path=model_path,
            output_dir="data/RCABENCH", 
            service_map_path="data/RCABENCH/service_to_idx.pkl",
            batch_size=32
        )
        answers = [
            AlgorithmAnswer(level="service", name=name, rank=i + 1)
            for i, name in enumerate(results)
        ]
        return answers



if __name__ == "__main__":
    registry = global_algorithm_registry()
    registry["padi-rca"] = PadiRCA
    main(enable_builtin_algorithms=False)