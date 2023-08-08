cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
set(ARMA_USE_WRAPPER true)

project(NTF LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NTFLIB_COMMON ${NTF_SOURCE_DIR}/../common)
set(NNLS_SRC ${NTF_SOURCE_DIR}/../nnls/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mbmi2") # for bit operations for ALTO

message(STATUS "NTF_SOURCE_DIR =  ${NTF_SOURCE_DIR}")
message(STATUS "NTFLIB_COMMON =  ${NTFLIB_COMMON}")

include(${NTFLIB_COMMON}/cmake_aux/NMFLIB_FindARMA.cmake)

message(STATUS "CMAKE_CXX_FLAGS           = ${CMAKE_CXX_FLAGS}"          )
message(STATUS "CMAKE_SHARED_LINKER_FLAGS = ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_REQUIRED_INCLUDES   = ${CMAKE_REQUIRED_INCLUDES}"  )

message(STATUS "*** libraries:")
message(STATUS "*** NTFLIB_LIBS = ${NTFLIB_LIBS} ${NMFLIB_LIBS}")

# set(CUDA_INCLUDE_DIRS ${NTF_SOURCE_DIR}/blcotensor_gpu)
# set(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)
# set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

# add_definitions(-fopenmp)

include_directories(
  ${ARMADILLO_INCLUDE_DIR}
  ${ARMADILLO_INCLUDE_DIRS}
  ${NTF_SOURCE_DIR}
  ${NTF_SOURCE_DIR}/../
  ${NTFLIB_COMMON}
  ${NNLS_SRC}
  ${Boost_INCLUDE_DIR}
  ${MKL_INCLUDE_DIR}
  ${OPENBLAS_INCLUDE_DIR}
  # ${CUDA_INCLUDE_DIRS}
  ${NTF_SOURCE_DIR}/blcotensor_gpu/include
)

set(CMAKE_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

set(CUDA_SOURCES 
  blcotensor_gpu/blco.cu
  blcotensor_gpu/cuda_utils.cu
  )  # set(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)

add_executable(sparse_ntf
sparse_ntf.cpp
)

add_library(blco_gpu ${CUDA_SOURCES})

# target_include_directories(sparse_ntf PRIVATE ${CUDA_INCLUDE_DIRS})

# target_compile_features(sparse_ntf PRIVATE cxx_std_17)
# target_link_libraries(sparse_ntf PRIVATE ${NTF_SOURCE_DIR}/blcotensor_gpu)
# target_link_directories(sparse_ntf PRIVATE ${NTF_SOURCE_DIR}/blcotensor_gpu)
target_compile_options(sparse_ntf PRIVATE
  # -march=native
  # -static
  -fopenmp
  # -O3
  # -D_GLIBCXX_PARALLEL
)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
target_include_directories(blco_gpu PRIVATE ${NTF_SOURCE_DIR}/blcotensor_gpu/include)

# Optionally, link any other necessary libraries
target_link_libraries(blco_gpu PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(sparse_ntf ${NTFLIB_LIBS} ${NMFLIB_LIBS} blco_gpu)

install(TARGETS sparse_ntf
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )
