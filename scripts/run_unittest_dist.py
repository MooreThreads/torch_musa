"""Script for running unittests concurrently."""

import argparse
import os
import subprocess
import threading
from os.path import join, dirname, split

import torch_musa

TORCH_MUSA_HOME = dirname(dirname(__file__))
DEFAULT_TEST_DIR = join(TORCH_MUSA_HOME, "tests/unittest")
TEST_REPORT_DIR = join(TORCH_MUSA_HOME, "build/reports/unit_test")


class UnittestParallelRunner:
    """Class definition for unittest runner."""

    def __init__(self, root_dir: str, report_dir: str, gpu_type: str):
        self.root_dir = root_dir
        self.gpu_groups = self.get_gpu_groups()
        self.gpu_type = gpu_type
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    def get_gpu_groups(self):
        gpu_counts = torch_musa.device_count()
        # Our GPU device has even number of devices except S80
        if gpu_counts == 1:
            return None
        return [[i, i + 1] for i in range(0, gpu_counts, 2)]

    def find_almost_all_tests(self):
        """Collect almost all tests except some cases need being taken care of."""
        all_tests = []
        for root, _, files in os.walk(self.root_dir):
            if split(root)[-1] in self.get_serial_tests():
                continue
            for file in files:
                if file.endswith(".py"):
                    all_tests.append(join(root, file))
        return all_tests

    def split_tasks(self):
        all_tests = self.find_almost_all_tests()
        groups_count = len(self.gpu_groups)
        tests_count = len(all_tests)
        each_tasks_amount = tests_count // groups_count
        return [
            all_tests[
                i * each_tasks_amount : min((i + 1) * each_tasks_amount, tests_count)
            ]
            for i in range(groups_count)
        ]

    def get_serial_tests(self):
        return ["profiler", "distributed", "core", "miscs"]

    def read_output(self, process):
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())

    def run_requires_serial(self, run_all=False):
        """Some unittests should be run serially."""
        serial_tests = []
        serial_tests.extend(self.get_serial_tests())
        serial_tests = [join(self.root_dir, x) for x in serial_tests]

        if run_all:
            serial_tests.extend(self.find_almost_all_tests())

        for serial_test in serial_tests:
            serial_test_str = serial_test
            if serial_test.endswith(".py"):
                serial_test_str = split(serial_test[:-3])[-1]

            pytest_command = (
                f"pytest --last-failed "
                f"--junitxml={self.report_dir}/{self.gpu_type}_{serial_test_str}_tests.xml"
            )
            print(f"Tests[{serial_test}] have been launched.")
            full_command = f"{pytest_command} {serial_test}"
            os.system(full_command)

    def run_on_multi_gpus(self):
        """Sufficiently utilize all gpus to run unittests."""
        threads = []
        processes = []
        test_groups = self.split_tasks()
        for (device_0, device_1), sub_tests in zip(self.gpu_groups, test_groups):
            sub_tests_str = "\n".join(sub_tests)
            test_list_name = f"{self.gpu_type}_{device_0}_{device_1}_test_list.txt"
            with open(test_list_name, "w", encoding="UTF-8") as f:
                f.write(sub_tests_str)

            pytest_command = (
                f"pytest --last-failed "
                f"--junitxml={self.report_dir}/{self.gpu_type}"
                f"_{device_0}_{device_1}_tests.xml"
            )
            full_command = (
                f"MUSA_VISIBLE_DEVICES={device_0},{device_1} "
                f"{pytest_command} @{test_list_name}"
            )
            # pylint: disable=consider-using-with
            process = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            print(f"Tests have been launched on gpu devices={device_0}_{device_1}.")
            threads.append(
                threading.Thread(
                    target=UnittestParallelRunner.read_output, args=(self, process)
                )
            )

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        flag = 0
        for p in processes:
            if p.stderr:
                p.stderr.close()
            if p.stdout:
                p.stdout.close()
            p.terminate()
            if not flag:
                flag = p.wait()
            else:
                p.wait()

    def run(self):
        if self.gpu_groups is None:
            self.run_requires_serial(run_all=True)
        else:
            self.run_requires_serial()
            self.run_on_multi_gpus()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test all unittests using all gpus")
    parser.add_argument(
        "--root-dir", help="Specify directory to be test", default=DEFAULT_TEST_DIR
    )
    parser.add_argument(
        "--report-dir",
        help="Specify directory to dump tests report",
        default=TEST_REPORT_DIR,
    )
    parser.add_argument("--gpu-type", help="Specify GPU type", default="S4000")
    args = parser.parse_args()
    tests_runner = UnittestParallelRunner(args.root_dir, args.report_dir, args.gpu_type)
    tests_runner.run()
