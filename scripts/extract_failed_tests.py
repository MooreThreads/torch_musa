"""Script for extracting failed tests form xml files of tests report."""

import xml.etree.ElementTree as ET
import sys


def filter_failed_tests():
    """filter failed tests from given tests report xml."""
    if len(sys.argv) < 2:
        return
    input_file = sys.argv[1]
    print(f"Extracting falied tests from {input_file}")
    tree = ET.parse(input_file)
    root = tree.getroot()
    new_testsuites = ET.Element("testsuites")

    for testsuite in root.findall("testsuite"):
        new_testsuite = ET.Element("testsuite", testsuite.attrib)
        for testcase in testsuite.findall("testcase"):
            if (
                testcase.find("failure") is not None
                or testcase.find("error") is not None
            ):
                new_testsuite.append(testcase)
        if len(new_testsuite) > 0:
            new_testsuites.append(new_testsuite)

    new_tree = ET.ElementTree(new_testsuites)
    new_tree.write(input_file, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    filter_failed_tests()
