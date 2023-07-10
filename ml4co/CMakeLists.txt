cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(ml4co LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native \
  -Wfatal-errors -fvisibility=hidden"
)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# PyTorch dependency
execute_process(
  COMMAND
  python -c
  "import os; import torch; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE
  TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
find_package(Torch REQUIRED)

# ml4co_ops lib
add_library(
  ml4co_ops
  SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cc/ml4co_ops.cc
)
target_include_directories(
  ml4co_ops
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(
  ml4co_ops
  PUBLIC
  torch
  split_and_pad_op
  prenorm_op
)


include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
  add_definitions(-DML4CO_CPU_ONLY)
endif()

macro(add_custom_op op_name)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_library(
      ${op_name}
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/cc/${op_name}.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/cc/${op_name}.cu
    )
    set_target_properties(
      ${op_name}
      PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
    )
  else()
    message(WARNING "[${op_name}] No CUDA complier, build cpu only version")
    add_library(
      ${op_name}
      ${CMAKE_CURRENT_SOURCE_DIR}/ops/cc/${op_name}.cc
    )
  endif()
  target_include_directories(
    ${op_name}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
  target_link_libraries(
    ${op_name}
    PUBLIC
    torch
  )
endmacro()

# Add custom ops lib
add_custom_op(split_and_pad_op)
add_custom_op(prenorm_op)
