import argparse

import pandas as pd


def match(cuda_op_name: str, mudnn_op_name: str):
    # e.g. log_softmax == LOGSOFTMAX, leak_relu_backward == LEAKY_RELU_BW
    return cuda_op_name.replace("_", "").replace("backward", "bw") == mudnn_op_name.lower().replace("_", "")


def check(cuda_ops_file_path: str, mudnn_ops_file_path: str):
    cuda_df = pd.read_excel(cuda_ops_file_path, "Sheet")
    mudnn_df = pd.read_excel(mudnn_ops_file_path, "Sheet")
    mudnn_op_names = mudnn_df["涉及算子的名称"]
    for index, row in cuda_df.iterrows():
        cuda_op_name = row["涉及算子的名称"]
        cuda_df.at[index, "muDNN是否支持"] = "N"
        for mudnn_op_name in mudnn_op_names:
            if match(cuda_op_name, mudnn_op_name):
                cuda_df.at[index, "muDNN是否支持"] = "Y"
                break

    cuda_df.to_excel(args.output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check ops between cuda and mudnn")
    parser.add_argument("--cuda-ops-file-path", help="Specify directory to be scanned", default="./cuda_full_ops.xlsx")
    parser.add_argument("--mudnn-ops-file-path", help="Specify file extension", default="./mudnn_support_ops.xlsx")
    parser.add_argument("--output-path", help="Specify ops' check result location",
                        default="./check_cuda_ops_in_mudnn.xlsx")

    args = parser.parse_args()
    test_cuda_ops_file_path = args.cuda_ops_file_path
    test_mudnn_ops_file_path = args.mudnn_ops_file_path
    check(test_cuda_ops_file_path, test_mudnn_ops_file_path)
