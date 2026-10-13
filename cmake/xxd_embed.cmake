foreach(required XXD INPUT SYMBOL OUTPUT)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing ${required} for xxd embedding")
    endif()
endforeach()
# 从 stdin 读取时 xxd -i 只输出字节列表，兼容没有 -n 参数的旧版 xxd。
execute_process(
    COMMAND "${XXD}" -i
    INPUT_FILE "${INPUT}"
    OUTPUT_VARIABLE embedded_bytes
    RESULT_VARIABLE xxd_result
    ERROR_VARIABLE xxd_error
)
if(NOT xxd_result STREQUAL "0")
    file(REMOVE "${OUTPUT}")
    message(FATAL_ERROR "xxd failed: ${xxd_error}")
endif()
file(WRITE "${OUTPUT}"
    "unsigned char ${SYMBOL}[] = {\n${embedded_bytes}};\n"
    "unsigned int ${SYMBOL}_len = sizeof(${SYMBOL});\n")
