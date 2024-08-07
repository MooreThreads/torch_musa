from tests import (  # noqa: F401  # noqa: F401
    binary_test,
    bmm_test,
    conv_test,
    linear_test,
    matmul_test,
    rmsnorm_test,
    unary_test,  # noqa: F401
    activation_test,
    gather_test,
    norm_test,
    shape_test,
    softmax_test,
    
)

import operator_benchmark as op_bench

if __name__ == "__main__":
    op_bench.benchmark_runner.main()
