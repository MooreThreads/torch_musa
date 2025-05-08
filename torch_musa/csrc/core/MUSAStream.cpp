#include "torch_musa/csrc/core/MUSAStream.h"

#include <array>
#include <atomic>

#include <c10/util/CallOnce.h>

#include "torch_musa/csrc/core/MUSAGuard.h"

namespace c10::musa {

namespace {

// Global stream state and constants
c10::once_flag init_flag;
DeviceIndex num_mtgpus = -1;
constexpr int kStreamsPerPoolBits = 5;
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
constexpr unsigned int kDefaultFlags = musaStreamNonBlocking;
constexpr int kStreamTypeBits = 4;

int max_stream_priorities;

std::array<c10::once_flag, MUSA_COMPILE_TIME_MAX_GPUS> device_flags;

std::array<
    std::array<std::atomic<uint32_t>, MUSA_COMPILE_TIME_MAX_GPUS>,
    max_compile_time_stream_priorities>
    priority_counters;

std::array<
    std::array<
        std::array<musaStream_t, kStreamsPerPool>,
        MUSA_COMPILE_TIME_MAX_GPUS>,
    max_compile_time_stream_priorities>
    streams;

class StreamIdType {
 private:
  uint8_t stream_type;

 public:
  static const uint8_t DEFAULT = 0x0;
  static const uint8_t EXT = 0xF;

 public:
  StreamIdType(const uint8_t _stream_type) : stream_type(_stream_type) {}

  bool isExt() const {
    return EXT == stream_type;
  }

  bool isDefault() const {
    return DEFAULT == stream_type;
  }

  uint8_t getStreamType() const {
    return stream_type;
  }
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  if (s.isDefault()) {
    stream << "DEFAULT";
  } else if (s.isExt()) {
    stream << "EXT";
  } else {
    stream << "PRIORITY " << static_cast<int>(s.getStreamType());
  }
  return stream;
}

StreamIdType streamIdType(StreamId s) {
  if ((!(s & 1)) && s) {
    return StreamIdType(StreamIdType::EXT);
  }

  constexpr int mask_for_type = (1 << kStreamTypeBits) - 1;
  const auto val = (s >> 1) & mask_for_type;
  TORCH_INTERNAL_ASSERT(val || !(s & 1), "invalid StreamId", s);
  return StreamIdType(val);
}

size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  if (st.isDefault()) {
    return static_cast<StreamId>(0);
  }
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      static_cast<StreamId>(st.getStreamType() << 1) | 1;
}

thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

void initGlobalStreamState() {
  num_mtgpus = device_count();
  TORCH_CHECK(
      num_mtgpus <= MUSA_COMPILE_TIME_MAX_GPUS,
      "Number of MUSA devices on the machine is larger than the compiled "
      "max number of mtgpus expected (",
      MUSA_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile.");

  const auto [leastPriority, greatestPriority] = MUSAStream::priority_range();
  max_stream_priorities = leastPriority - greatestPriority + 1;
}

void initDeviceStreamState(DeviceIndex device_index) {
  const MUSAGuard device_guard(device_index);

  auto initSingleStream = [device_index](int torch_pri, int i) {
    auto& stream = streams[torch_pri][device_index][i];
    const auto musa_pri = -torch_pri;

    C10_MUSA_CHECK(
        musaStreamCreateWithPriority(&stream, kDefaultFlags, musa_pri));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_stream_creation(
          kMUSA, reinterpret_cast<uintptr_t>(stream));
      priority_counters[torch_pri][device_index] = 0;
    }
  };

  for (const auto i : c10::irange(kStreamsPerPool)) {
    for (const auto p : c10::irange(max_stream_priorities)) {
      initSingleStream(p, i);
    }
  }
}

void initMUSAStreamsOnce() {
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  current_streams = std::make_unique<StreamId[]>(num_mtgpus);
  for (const auto i : c10::irange(num_mtgpus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

void check_gpu(DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < num_mtgpus);
}

// Note: Streams are returned round-robin.
uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

MUSAStream MUSAStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return MUSAStream(
      MUSAStream::UNCHECKED,
      Stream(Stream::UNSAFE, Device(kMUSA, device_index), stream_id));
}

} // anonymous namespace

musaStream_t MUSAStream::stream() const {
  DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  if (st.isDefault()) {
    TORCH_INTERNAL_ASSERT(
        si == 0,
        "Unrecognized stream ",
        stream_,
        " (I think this should be the default stream, but I got a non-zero index ",
        si,
        ").",
        " Did you manufacture the StreamId yourself?  Don't do that; use the",
        " official API like c10::musa::getStreamFromPool() to get a new stream.");
    return nullptr;
  } else if (st.isExt()) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<musaStream_t>(stream_id);
  } else {
    auto streamType = st.getStreamType();
    TORCH_INTERNAL_ASSERT(
        streamType >= 1 && streamType <= max_stream_priorities,
        "Unrecognized stream ",
        stream_,
        " (I didn't recognize the stream type, ",
        st,
        " with the value ",
        streamType,
        ")");
    return streams[st.getStreamType() - 1][device_index][si];
  }
}

MUSAStream getStreamFromPool(
    const bool isHighPriority,
    DeviceIndex device_index) {
  initMUSAStreamsOnce();
  const int priority = isHighPriority ? -max_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device_index);
}

MUSAStream getStreamFromPool(const int priority, DeviceIndex device_index) {
  initMUSAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    SetTargetDevice();
  }
  TORCH_CHECK(
      priority <= 0,
      "Expected musa stream priority to be less than or equal to 0, got ",
      priority);
  check_gpu(device_index);

  c10::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  auto pri_idx = -priority;
  pri_idx =
      std::min(pri_idx, max_stream_priorities - 1); // pri_idx is zero-based
  const auto idx = get_idx(priority_counters[pri_idx][device_index]);
  StreamIdType id_type = StreamIdType(pri_idx + 1);
  return MUSAStreamForId(device_index, makeStreamId(id_type, idx));
}

MUSAStream getStreamFromExternal(
    musaStream_t ext_stream,
    DeviceIndex device_index) {
  // The stream pointer will be the actual id
  return MUSAStreamForId(device_index, reinterpret_cast<int64_t>(ext_stream));
}

MUSAStream getDefaultMUSAStream(DeviceIndex device_index) {
  initMUSAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    SetTargetDevice();
  }
  check_gpu(device_index);
  return MUSAStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

MUSAStream getCurrentMUSAStream(DeviceIndex device_index) {
  initMUSAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
    SetTargetDevice();
  }
  check_gpu(device_index);
  return MUSAStreamForId(device_index, current_streams[device_index]);
}

void setCurrentMUSAStream(MUSAStream stream) {
  initMUSAStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const MUSAStream& s) {
  return stream << s.unwrap();
}

} // namespace c10::musa
