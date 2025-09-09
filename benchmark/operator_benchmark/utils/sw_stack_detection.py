import subprocess
import sys
import json
import re
import torch
import torch_musa


"""
Auxiliary tools for MUSA software version query.
"""


def execute_command(command, args=None):
    try:
        subprocess.check_output(["which", f"{command}"])
    except subprocess.CalledProcessError:
        print(f"Error: '{command}' command isn't located at the system path.")
        sys.exit(1)

    command = f"{command} {args}" if args is not None else command
    try:
        output_of_command = (
            subprocess.check_output([command], shell=True).decode().strip()
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: execute command: '{command} failed!'")
        raise e
    return output_of_command


def get_musart_version():
    command = "musa_runtime_version"
    output = execute_command(command)
    # sample:
    """
     musa_runtime:{"version":"4.1.0","gitbranch":"HEAD","gittag":"Notag","commitid":"1301a3d6b62f56f6fe0126be25f34e6aac0c95ff","commitdate":"2025-06-2317:39:48+0800"}
    """
    stripped_string = re.sub(r"[\n\t\s]*", "", output)
    # WARNING: This is highly related to MUSA Runtime version string, so keep
    # updating these hard-coded string extracted rightly with musa version changing.
    musa_runtime_str = stripped_string.replace("musa_runtime:", "")
    musa_runtime_info = json.loads(musa_runtime_str)
    return {
        "musa_runtime_info": musa_runtime_info,
    }


def get_musa_toolkits_version():
    command = "musa_toolkits_version"
    output = execute_command(command)
    stripped_string = re.sub(r"[\n\t\s]*", "", output)
    toolkits_str = stripped_string.replace("musa_toolkits:", "")
    toolkits_info = json.loads(toolkits_str)
    return toolkits_info


def get_mudnn_version():
    command = "mudnn_version"
    output = execute_command(command)
    stripped_string = re.sub(r"[\n\t\s]*", "", output)
    mudnn_version_str = stripped_string.replace("mudnn:", "")
    mudnn_version_info = json.loads(mudnn_version_str)
    return mudnn_version_info


def get_musa_stack_version():
    res = {}
    res["musa_runtime"] = get_musart_version()
    res["musa_toolkits"] = get_musa_toolkits_version()
    res["mudnn"] = get_mudnn_version()

    return res


if __name__ == "__main__":
    # test all
    print(get_musart_version())
    print(get_musa_stack_version())
    print(get_mudnn_version())
    print(get_musa_toolkits_version())
