cmake_minimum_required(VERSION 3.20)
project(kalmx LANGUAGES CXX)

include(cmake/KalmxOptions.cmake)

# option(KALMX_BUILD_PYBIND "Build Python bindings via pybind11" ON)
# option(KALMX_BUILD_BENCH   "Build benchmarks" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(kalmx
  src/kf.cpp
)
target_include_directories(kalmx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kalmx PUBLIC Eigen3::Eigen)
kalmx_set_warnings(kalmx)

add_executable(kf_demo examples/kf_demo.cpp)
target_link_libraries(kf_demo PRIVATE kalmx)

# enable_testing()
# add_executable(smoke_test tests/smoke_test.cpp)
# target_link_libraries(smoke_test PRIVATE kalmx)
# add_test(NAME smoke_test COMMAND smoke_test)

# if(KALMX_BUILD_BENCH)
#   add_executable(bench_kf benchmarks/bench_kf.cpp)
#   target_link_libraries(bench_kf PRIVATE kalmx)
# endif()

# if(KALMX_BUILD_PYBIND)
#   find_package(pybind11 QUIET)
#   if(pybind11_FOUND)
#     pybind11_add_module(kalmx_py bindings/pybind_module.cpp)
#     target_link_libraries(kalmx_py PRIVATE kalmx)
#   else()
#     message(STATUS "pybind11 not found — Python module will be skipped")
#   endif()
# endif()

# Install (headers + lib)
include(GNUInstallDirs)
install(TARGETS kalmx
  EXPORT kalmxTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
