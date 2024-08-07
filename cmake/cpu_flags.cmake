# Misc check from PyTorch, some flags could be used by torch_musa to keep
# consistent with PyTorch.
include(${BUILD_PYTORCH_REPO_PATH}/cmake/MiscCheck.cmake)

list(APPEND CPU_CAPABILITY_NAMES "DEFAULT")

# TODO:(mt-ai) OPT_FLAG may need to be defined in the future.
list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}")

if(CXX_AVX512_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX512_CPU_DEFINITION")
  list(APPEND CPU_CAPABILITY_NAMES "AVX512")

  if(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512")
  else(MSVC)
    list(APPEND CPU_CAPABILITY_FLAGS
         "${OPT_FLAG} -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma")
  endif(MSVC)
endif(CXX_AVX512_FOUND)

if(CXX_AVX2_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_AVX2_CPU_DEFINITION")

  # Some versions of GCC pessimistically split unaligned load and store
  # instructions when using the default tuning. This is a bad choice on new
  # Intel and AMD processors so we disable it when compiling with AVX2. See
  # https://stackoverflow.com/questions/52626726/why-doesnt-gcc-resolve-mm256-loadu-pd-as-single-vmovupd#tab-top
  check_cxx_compiler_flag(
    "-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store"
    COMPILER_SUPPORTS_NO_AVX256_SPLIT)

  if(COMPILER_SUPPORTS_NO_AVX256_SPLIT)
    set(CPU_NO_AVX256_SPLIT_FLAGS
        "-mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store")
  endif(COMPILER_SUPPORTS_NO_AVX256_SPLIT)

  list(APPEND CPU_CAPABILITY_NAMES "AVX2")

  if(DEFINED ENV{ATEN_AVX512_256})
    if($ENV{ATEN_AVX512_256} MATCHES "TRUE")
      if(CXX_AVX512_FOUND)
        message("-- ATen AVX2 kernels will use 32 ymm registers")

        if(MSVC)
          list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX512")
        else(MSVC)
          list(APPEND CPU_CAPABILITY_FLAGS
               "${OPT_FLAG} -march=native ${CPU_NO_AVX256_SPLIT_FLAGS}")
        endif(MSVC)
      endif(CXX_AVX512_FOUND)
    endif()
  else()
    if(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS "${OPT_FLAG}/arch:AVX2")
    else(MSVC)
      list(APPEND CPU_CAPABILITY_FLAGS
           "${OPT_FLAG} -mavx2 -mfma ${CPU_NO_AVX256_SPLIT_FLAGS}")
    endif(MSVC)
  endif()
endif(CXX_AVX2_FOUND)

# WARNING: From the above, now only AVX platform flags set in torch_musa. It's
# sure that there will be more platforms integrated, but now TEMPORARILY we just
# admit these two ISAs, lol.
