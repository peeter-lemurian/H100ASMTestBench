add_executable(fp32TObf16 fp32TObf16.cu)

set_target_properties(fp32TObf16 PROPERTIES OUTPUT_NAME fp32TObf16)

add_executable(bf16TOfp32 bf16TOfp32.cu)

set_target_properties(bf16TOfp32 PROPERTIES OUTPUT_NAME bf16TOfp32)
