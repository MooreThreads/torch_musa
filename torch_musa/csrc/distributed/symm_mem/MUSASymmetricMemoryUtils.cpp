#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <c10/util/error.h>

#include "torch_musa/csrc/core/driver_api.h"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemoryUtils.hpp"

namespace c10d::symmetric_memory {

bool device_has_multicast_support(int device_idx) {
  (void)device_idx;
  return false;
}

IpcChannel::IpcChannel()
    : socket_name_(get_socket_name(getpid())),
      socket_(socket(AF_UNIX, SOCK_DGRAM, 0)) {
  TORCH_CHECK(
      socket_ != -1, "Failed to create socket: ", c10::utils::str_error(errno));

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  TORCH_CHECK(
      bind(socket_, (struct sockaddr*)&addr, SUN_LEN(&addr)) == 0,
      "Failed to bind socket: ",
      c10::utils::str_error(errno));
}

IpcChannel::~IpcChannel() {
  close(socket_);
  unlink(socket_name_.c_str());
}

void IpcChannel::send_fd(int dst_pid, int fd) {
  // Because file descriptors are process-local kernel objects, and we can’t
  // pass them via normal socket payloads (like write() or send()).  Unix domain
  // sockets provide a mechanism to pass actual FDs via sendmsg()/recvmsg().
  // Define destination socket address
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  auto socket_name = get_socket_name(dst_pid);
  std::copy(socket_name.begin(), socket_name.end(), addr.sun_path);

  struct iovec io = {.iov_base = (void*)"fd", .iov_len = 2};
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Create message header
  struct msghdr msg {
    // destination socket address and size of it
    // message content in msg_iov and number of such structs (1 in our case)
    // auxiliary data with the value of fd and size of it
    .msg_name = (void*)&addr, .msg_namelen = sizeof(struct sockaddr_un),
    .msg_iov = &io, .msg_iovlen = 1, .msg_control = cbuf,
    .msg_controllen = sizeof(cbuf)
  };

  // This points to the first control message header
  // With SCM_RIGHTS we let the kernel know that we are passing file
  // descriptors.
  auto cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  // Specify socket level message
  cmsg->cmsg_level = SOL_SOCKET;
  // SCM_RIGHTS is the type used to pass file descriptors
  cmsg->cmsg_type = SCM_RIGHTS;

  if (fd != -1) {
    std::copy(
        reinterpret_cast<const char*>(&fd),
        reinterpret_cast<const char*>(&fd) + sizeof(fd),
        reinterpret_cast<char*>(CMSG_DATA(cmsg)));
  } else {
    msg.msg_controllen = 0;
  }

  // Finally send the message
  TORCH_CHECK(
      sendmsg(socket_, &msg, 0) > 0,
      "Failed to send fd: ",
      c10::utils::str_error(errno));
}

int IpcChannel::recv_fd() {
  // Prepare buffer for regular message "fd"
  // NOLINTNEXTLINE(*array*)
  char buf[2];
  memset(&buf, 0, sizeof(buf));
  struct iovec io = {.iov_base = (void*)buf, .iov_len = sizeof(buf)};

  // Prepare buffer for control message and zero it out
  // NOLINTNEXTLINE(*array*)
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Define socket address to receive on: family AF_UNIX means unix domain
  // socket
  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  std::copy(socket_name_.begin(), socket_name_.end(), addr.sun_path);

  // Prepare message header
  struct msghdr msg = {
      .msg_name = (void*)&addr,
      .msg_namelen = sizeof(struct sockaddr_un),
      .msg_iov = &io,
      .msg_iovlen = 1,
      .msg_control = cbuf,
      .msg_controllen = sizeof(cbuf)};

  // Receive message on socket_
  TORCH_CHECK(
      recvmsg(socket_, &msg, 0) > 0,
      "Failed to receive fd: ",
      c10::utils::str_error(errno));

  if (msg.msg_controllen == 0) {
    return -1;
  }

  // Extract control message and validate its content
  auto cmsg = CMSG_FIRSTHDR(&msg);
  TORCH_CHECK(cmsg != nullptr);
  TORCH_CHECK(cmsg->cmsg_len == CMSG_LEN(sizeof(int)));
  TORCH_CHECK(cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS);
  return *reinterpret_cast<int*>(CMSG_DATA(cmsg));
}

std::vector<int> IpcChannel::all_gather_fds(
    int rank,
    const std::vector<int>& pids,
    int fd) {
  int world_size = static_cast<int>(pids.size());
  std::vector<int> fds(pids.size());
  fds[rank] = fd;

  int dst_rank = (rank + 1) % world_size;
  for (int step = 1; step < world_size; ++step) {
    int src_rank = (rank + world_size - step) % world_size;
    send_fd(pids[dst_rank], fd);
    fd = recv_fd();
    fds[src_rank] = fd;
  }
  return fds;
}

int IpcChannel::broadcast_fds(
    int rank,
    int src_rank,
    const std::vector<int>& pids,
    int fd) {
  int world_size = static_cast<int>(pids.size());

  if (rank == src_rank) {
    for (int dst_rank = 0; dst_rank < world_size; ++dst_rank) {
      if (dst_rank == rank) {
        continue;
      }
      send_fd(pids[dst_rank], fd);
    }
    return fd;
  }
  return recv_fd();
}

std::string IpcChannel::get_socket_name(int pid) {
  const char* tmp_dir = "/tmp";
  for (const char* env_var : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    if (const char* path = getenv(env_var)) {
      tmp_dir = path;
      break;
    }
  }
  std::ostringstream oss;
  oss << tmp_dir << "/symm_mem-" << pid;
  return oss.str();
}

void map_block(void** ptr, HandleType handle, size_t size, int device_idx) {
  auto driver_api = c10::musa::DriverAPI::get();
  auto dev_ptr = reinterpret_cast<MUdeviceptr*>(ptr);
  C10_MUSA_DRIVER_CHECK(
      driver_api->muMemAddressReserve_(dev_ptr, size, 0ULL, 0, 0ULL));
  C10_MUSA_DRIVER_CHECK(driver_api->muMemMap_(*dev_ptr, size, 0, handle, 0ULL));

  MUmemAccessDesc desc{};
  desc.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device_idx;
  desc.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  C10_MUSA_DRIVER_CHECK(driver_api->muMemSetAccess_(*dev_ptr, size, &desc, 1));
}

} // namespace c10d::symmetric_memory
