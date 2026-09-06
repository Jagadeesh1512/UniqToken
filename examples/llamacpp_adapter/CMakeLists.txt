cmake_minimum_required(VERSION 3.14)
project(uniqtoken_llamacpp_adapter C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(uniqtoken_llama uniqtoken_llama.cpp)
target_include_directories(uniqtoken_llama PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Link against uniqtoken_core library using absolute paths relative to the
# current source directory so out-of-source builds resolve correctly.
find_library(
  UNIQTOKEN_CORE_LIB
  uniqtoken_core
  PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../target/release
    ${CMAKE_CURRENT_SOURCE_DIR}/../../target/debug
    ${CMAKE_CURRENT_SOURCE_DIR}/../../crates/uniqtoken_core/target/release
    ${CMAKE_CURRENT_SOURCE_DIR}/../../crates/uniqtoken_core/target/debug
  NO_DEFAULT_PATH
)
if (NOT UNIQTOKEN_CORE_LIB)
  message(FATAL_ERROR "uniqtoken_core library not found. Please build the Rust crate first: cargo build --manifest-path crates/uniqtoken_core/Cargo.toml --release --no-default-features --features c_abi")
endif()
target_link_libraries(uniqtoken_llama PRIVATE ${UNIQTOKEN_CORE_LIB})
