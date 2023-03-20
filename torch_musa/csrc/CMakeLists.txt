FILE(GLOB _CSRCS
    aten/ops/*.cpp
    aten/utils/*.cpp
    c10/*.cpp
    )

LIST(APPEND MUSA_CSRCS ${_CSRCS})

# Pass to parent
set(MUSA_CSRCS ${MUSA_CSRCS} PARENT_SCOPE)
