cmake_minimum_required(VERSION 3.20)
project(qhmat VERSION 0.1.0 LANGUAGES CXX)

option(QHMAT_ENABLE_CUDA "Build the optional CUDA backend" OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_library(qhmat_core STATIC src/qhmat/core.cpp src/qhmat/io.cpp)
target_compile_features(qhmat_core PUBLIC cxx_std_17)
target_include_directories(qhmat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(qhmat_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(_core src/qhmat/bindings.cpp)
target_link_libraries(_core PRIVATE qhmat_core)
target_compile_features(_core PRIVATE cxx_std_17)
install(TARGETS _core LIBRARY DESTINATION qhmat)

if(QHMAT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(qhmat_cuda STATIC src/qhmat/cuda.cu)
  target_include_directories(qhmat_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(qhmat_cuda PUBLIC CUDA::cudart)
  target_compile_features(qhmat_cuda PUBLIC cuda_std_17)
  set_target_properties(qhmat_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(_core PRIVATE qhmat_cuda)
  target_compile_definitions(_core PRIVATE QHMAT_HAS_CUDA=1)
endif()
