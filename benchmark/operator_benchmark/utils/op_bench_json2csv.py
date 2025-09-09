import os
import argparse
import pandas as pd
import json
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Tool to visualize the result of operator benchmark in JSON format.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


def parse_args():
    parser.add_argument(
        "res_file", help="file path of the saved result of op benchmark."
    )
    args, _ = parser.parse_known_args()
    return args


def visualize_json(json_file_path):
    with open(json_file_path, "r") as f:
        res = json.load(f)
    result_list = res["benchmark_result"]
    cols_name = [
        "OP",
        "Mode",
        "TestConfig",  # str
        "backward",
        "intensity",
        "memory",
        "gb/s",
        "FLOPS",
        "TFLOPs",
        "lat_mean",
        "lat_variance",
        "0%",
        "25%",
        "50%",
        "75%",
        "100%",
    ]

    res_df = pd.DataFrame(columns=cols_name)
    for op_res in result_list:
        if not op_res:
            continue
        op_name = op_res["op"]
        mode = op_res["mode"]
        for test_case in op_res["test_cases"]:
            row = [op_name, mode]
            row.append(str(test_case["test_config"]))
            row.append(test_case["backward"])
            row.append(test_case["intensity"])
            memory = int(test_case["memory"])
            if memory >= 1e9:
                memory = str(round(memory / 1e9, 3)) + "GB"
            else:
                memory = str(round(memory / 1e6, 1)) + "MB"
            row.append(memory)
            row.append(test_case["gb/s"])
            flops = int(test_case["flops"])
            if flops >= 1e10:
                flops = str(round(flops / 1e12, 3)) + "T"
            else:
                flops = str(round(flops / 1e9, 3)) + "G"
            row.append(flops)
            row.append(test_case["tflops"])
            time_metric = test_case["time_metric"]
            row.append(time_metric[0])
            row.append(time_metric[1])
            for percent in time_metric[2]:
                row.append(percent)

            res_df.loc[len(res_df)] = row

    json_file_name = Path(json_file_path).stem

    res_df.to_csv(
        os.path.join(os.path.dirname(json_file_path), f"{json_file_name}.csv"),
        index=False,
    )


if __name__ == "__main__":
    args = parse_args()
    json_file_path = args.res_file
    visualize_json(json_file_path)
