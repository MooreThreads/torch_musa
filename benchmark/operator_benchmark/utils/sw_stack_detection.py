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
    "musa_runtime:\n{\n\t\"
     version\":\t\"2.1.0\",\n\t\"git branch\":\t\"rc2.1.0\",\n\t\"git tag\":\t\"No tag\",\n\t\"commit id\":\t\"871ff3c18bb06e3c521275b0e7732b674cddb6dd\",\n\t\"commit date\":\t\"2024-03-25 15:29:03 +0800\"\n}\ndriver_dependency:\n{\n\t\"git branch\":\t\"heads/20240320_develop\",\n\t\"git tag\":\t\"20240320_develop\",\n\t\"commit id\":\t\"4f591d074070595db19003e5069e78a7ff8942d1\",\n\t\"commit date\":\t\"2024-03-20 15:28:45 +0800\"\n}"
    """
    stripped_string = re.sub(r"[\n\t\s]*", "", output)
    # WARNING: This is highly related to MUSA Runtime version string, so keep
    # updating these hard-coded string extracted rightly with musa version changing.
    index_of_driver = stripped_string.find("driver_dependency")
    musa_runtime_str = stripped_string[:index_of_driver].replace("musa_runtime:", "")
    driver_dependency_str = stripped_string[index_of_driver:].replace(
        "driver_dependency:", ""
    )
    musa_runtime_info = json.loads(musa_runtime_str)
    driver_depends_info = json.loads(driver_dependency_str)

    return {
        "musa_runtime_info": musa_runtime_info,
        "driver_depends_info": driver_depends_info,
    }


def get_ddk_version():
    command = "clinfo"
    output = execute_command(command=command, args="| grep Driver")
    output = output.splitlines()[
        0
    ]  # Only get the ddk version of the first device card.
    output = re.sub(r"\s+", " ", output).strip()
    _, _, ddk_date, ddk_type, ddk_version, ddk_commit = output.split(" ")

    return {
        "ddk_date": ddk_date,
        "ddk_type": ddk_type,
        "ddk_version": ddk_version,
        "ddk_commit": ddk_commit,
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
    print(get_ddk_version())
    print(get_musart_version())
    print(get_musa_stack_version())
    print(get_mudnn_version())
    print(get_musa_toolkits_version())
