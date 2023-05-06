#include "torch_musa/csrc/core/MUSAStream.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace c10 {
namespace musa {
namespace {
// Global stream state and constants
static c10::once_flag init_flag;
static DeviceIndex num_mtgpus = -1;
static constexpr int kStreamsPerPoolBits = 5;
static constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
static constexpr unsigned int kDefaultFlags = musaStreamNonBlocking;
static constexpr int kStreamTypeBits = 3;

// Note: lower numbers are higher priorities, zero is default priority
static constexpr int kHighPriority = 0;
static constexpr int kLowPriority = 1;

static c10::once_flag device_flags[MUSA_COMPILE_TIME_MAX_GPUS];
static std::atomic<uint32_t> low_priority_counters[MUSA_COMPILE_TIME_MAX_GPUS];
static std::atomic<uint32_t> high_priority_counters[MUSA_COMPILE_TIME_MAX_GPUS];
static musaStream_t low_priority_streams[MUSA_COMPILE_TIME_MAX_GPUS]
                                        [kStreamsPerPool];
static musaStream_t high_priority_streams[MUSA_COMPILE_TIME_MAX_GPUS]
                                         [kStreamsPerPool];

enum class StreamIdType : uint8_t {
  DEFAULT = 0x0,
  LOW = 0x1,
  HIGH = 0x2,
  EXT = 0x3,
};

std::ostream& operator<<(std::ostream& stream, StreamIdType s) {
  switch (s) {
    case StreamIdType::DEFAULT:
      stream << "DEFAULT";
      break;
    case StreamIdType::LOW:
      stream << "LOW";
      break;
    case StreamIdType::HIGH:
      stream << "HIGH";
      break;
    case StreamIdType::EXT:
      stream << "EXT";
      break;
    default:
      stream << static_cast<uint8_t>(s);
      break;
  }
  return stream;
}

static inline StreamIdType streamIdType(StreamId s) {
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  if (s && ((s & mask_for_type) == 0)) {
    // Externally allocated streams have their id being the musaStream_ptr
    // so the bits corresponding to the type will be 0 and will collide with
    // the default stream.
    return StreamIdType::EXT;
  }
  return static_cast<StreamIdType>(s & mask_for_type);
}

static inline size_t streamIdIndex(StreamId s) {
  return static_cast<size_t>(
      (s >> kStreamTypeBits) & ((1 << kStreamsPerPoolBits) - 1));
}

StreamId makeStreamId(StreamIdType st, size_t si) {
  return (static_cast<StreamId>(si) << kStreamTypeBits) |
      static_cast<StreamId>(st);
}

// Thread-local current streams
static thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// Warning: this function must only be called once!
static void initGlobalStreamState() {
  num_mtgpus = device_count();
  // Check if the number of GPUs matches the expected compile-time max number
  // of Moore Threads GPUs.
  TORCH_CHECK(
      num_mtgpus <= MUSA_COMPILE_TIME_MAX_GPUS,
      "Number of MUSA devices on the machine is larger than the compiled "
      "max number of mtgpus expected (",
      MUSA_COMPILE_TIME_MAX_GPUS,
      "). Increase that and recompile.");
}

// Creates the low and high priority stream pools for the specified device
// Warning: only call once per device!
static void initDeviceStreamState(DeviceIndex device_index) {
  // Switches to the requested device so streams are properly associated
  // with it.
  MUSAGuard device_guard(device_index);

  for (const auto i : c10::irange(kStreamsPerPool)) {
    auto& lowpri_stream = low_priority_streams[device_index][i];
    auto& hipri_stream = high_priority_streams[device_index][i];

    TORCH_MUSA_CHECK(musaStreamCreateWithPriority(
        &lowpri_stream, kDefaultFlags, kLowPriority));
    TORCH_MUSA_CHECK(musaStreamCreateWithPriority(
        &hipri_stream, kDefaultFlags, kHighPriority));
  }

  low_priority_counters[device_index] = 0;
  high_priority_counters[device_index] = 0;
}

// Init front-end to ensure initialization only occurs once
static void initMUSAStreamsOnce() {
  // Inits default streams (once, globally)
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to default streams
  current_streams = std::make_unique<StreamId[]>(num_mtgpus);
  for (const auto i : c10::irange(num_mtgpus)) {
    current_streams[i] = makeStreamId(StreamIdType::DEFAULT, 0);
  }
}

// Helper to verify the Moore Threads GPU index is valid.
static inline void check_gpu(DeviceIndex device_index) {
  TORCH_INTERNAL_ASSERT(device_index >= 0 && device_index < num_mtgpus);
}

// Helper to determine the index of the stream to return.
// Note: Streams are returned round-robin.
static uint32_t get_idx(std::atomic<uint32_t>& counter) {
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
  c10::StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  size_t si = streamIdIndex(stream_id);
  switch (st) {
    case StreamIdType::DEFAULT:
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
    case StreamIdType::LOW:
      return low_priority_streams[device_index][si];
    case StreamIdType::HIGH:
      return high_priority_streams[device_index][si];
    case StreamIdType::EXT:
      return reinterpret_cast<musaStream_t>(stream_id);
    default:
      TORCH_INTERNAL_ASSERT(
          0,
          "Unrecognized stream ",
          stream_,
          " (I didn't recognize the stream type, ",
          st,
          ")");
  }
}

MUSAStream getStreamFromPool(
    const bool isHighPriority,
    DeviceIndex device_index) {
  initMUSAStreamsOnce();
  if (device_index == -1)
    device_index = current_device();
  check_gpu(device_index);

  // Initializes the stream pools (once)
  c10::call_once(
      device_flags[device_index], initDeviceStreamState, device_index);

  if (isHighPriority) {
    const auto idx = get_idx(high_priority_counters[device_index]);
    return MUSAStreamForId(device_index, makeStreamId(StreamIdType::HIGH, idx));
  }

  const auto idx = get_idx(low_priority_counters[device_index]);
  return MUSAStreamForId(device_index, makeStreamId(StreamIdType::LOW, idx));
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
  }
  check_gpu(device_index);
  return MUSAStreamForId(device_index, makeStreamId(StreamIdType::DEFAULT, 0));
}

MUSAStream getCurrentMUSAStream(DeviceIndex device_index) {
  initMUSAStreamsOnce();
  if (device_index == -1) {
    device_index = current_device();
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

} // namespace musa
} // namespace c10
