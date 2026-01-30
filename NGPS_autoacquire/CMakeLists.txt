cmake_minimum_required(VERSION 3.16)
project(ngps_acq C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(ngps_acq src/ngps_acq.c)

find_package(PkgConfig REQUIRED)

pkg_check_modules(CFITSIO REQUIRED cfitsio)
pkg_check_modules(WCSLIB REQUIRED wcslib)

target_include_directories(ngps_acq PRIVATE
  ${CFITSIO_INCLUDE_DIRS}
  ${WCSLIB_INCLUDE_DIRS}
)

target_link_directories(ngps_acq PRIVATE
  ${CFITSIO_LIBRARY_DIRS}
  ${WCSLIB_LIBRARY_DIRS}
)

target_link_libraries(ngps_acq PRIVATE
  ${CFITSIO_LIBRARIES}
  ${WCSLIB_LIBRARIES}
  m
)
