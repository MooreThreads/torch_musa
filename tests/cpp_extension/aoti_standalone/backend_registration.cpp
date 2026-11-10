#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Exception.h>

#include <fstream>
#include <iostream>
#include <string>

int main() {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "aten::empty", "memory_format");
  TORCH_CHECK(
      op.hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1),
      "MUSA backend registrations were discarded by the linker");

  std::ifstream maps("/proc/self/maps");
  TORCH_CHECK(maps.is_open(), "Unable to inspect process mappings");
  std::string line;
  while (std::getline(maps, line)) {
    TORCH_CHECK(
        line.find("libpython") == std::string::npos &&
            line.find("libtorch_python") == std::string::npos &&
            line.find("libmusa_python") == std::string::npos,
        "Python component loaded in the standalone process: ",
        line);
  }
  std::cout
      << "PASS: standalone MUSA backend registered; no Python libraries\n";
}
