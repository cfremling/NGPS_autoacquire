cmake_minimum_required(VERSION 3.16)
project(ngps_acq C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(ngps_acq src/ngps_acq.c)

# --- Prefer pkg-config if available, but do NOT require it ---
find_package(PkgConfig QUIET)

# ---------------- CFITSIO ----------------
set(CFITSIO_FOUND FALSE)

if(PkgConfig_FOUND)
  pkg_check_modules(PC_CFITSIO QUIET cfitsio)
  if(PC_CFITSIO_FOUND)
    set(CFITSIO_FOUND TRUE)
    set(CFITSIO_INCLUDE_DIRS ${PC_CFITSIO_INCLUDE_DIRS})
    set(CFITSIO_LIBRARY_DIRS ${PC_CFITSIO_LIBRARY_DIRS})
    set(CFITSIO_LIBRARIES    ${PC_CFITSIO_LIBRARIES})
  endif()
endif()

if(NOT CFITSIO_FOUND)
  find_path(CFITSIO_INCLUDE_DIR
    NAMES fitsio.h
    PATHS
      /usr/local/include
      /usr/include
      /opt/homebrew/include
      /opt/local/include
  )
  find_library(CFITSIO_LIBRARY
    NAMES cfitsio
    PATHS
      /usr/local/lib
      /usr/lib
      /usr/lib/x86_64-linux-gnu
      /opt/homebrew/lib
      /opt/local/lib
  )

  if(CFITSIO_INCLUDE_DIR AND CFITSIO_LIBRARY)
    set(CFITSIO_FOUND TRUE)
    set(CFITSIO_INCLUDE_DIRS ${CFITSIO_INCLUDE_DIR})
    set(CFITSIO_LIBRARIES    ${CFITSIO_LIBRARY})
  endif()
endif()

# ---------------- WCSLIB ----------------
set(WCSLIB_FOUND FALSE)

if(PkgConfig_FOUND)
  pkg_check_modules(PC_WCSLIB QUIET wcslib)
  if(PC_WCSLIB_FOUND)
    set(WCSLIB_FOUND TRUE)
    set(WCSLIB_INCLUDE_DIRS ${PC_WCSLIB_INCLUDE_DIRS})
    set(WCSLIB_LIBRARY_DIRS ${PC_WCSLIB_LIBRARY_DIRS})
    set(WCSLIB_LIBRARIES    ${PC_WCSLIB_LIBRARIES})
  endif()
endif()

if(NOT WCSLIB_FOUND)
  # Common header layouts:
  #   /usr/include/wcslib/wcs.h     (preferred for #include <wcslib/wcs.h>)
  #   /usr/include/wcslib/wcshdr.h
  find_path(WCSLIB_INCLUDE_DIR
    NAMES wcslib/wcs.h wcslib/wcshdr.h
    PATHS
      /usr/local/include
      /usr/include
      /opt/homebrew/include
      /opt/local/include
  )

  # Fallback: some installs put headers directly under .../include/wcslib/
  if(NOT WCSLIB_INCLUDE_DIR)
    find_path(WCSLIB_INCLUDE_DIR
      NAMES wcs.h wcshdr.h
      PATHS
        /usr/local/include/wcslib
        /usr/include/wcslib
        /opt/homebrew/include/wcslib
        /opt/local/include/wcslib
    )
  endif()

  find_library(WCSLIB_LIBRARY
    NAMES wcs
    PATHS
      /usr/local/lib
      /usr/lib
      /usr/lib/x86_64-linux-gnu
      /opt/homebrew/lib
      /opt/local/lib
  )

  if(WCSLIB_INCLUDE_DIR AND WCSLIB_LIBRARY)
    set(WCSLIB_FOUND TRUE)
    set(WCSLIB_INCLUDE_DIRS ${WCSLIB_INCLUDE_DIR})
    set(WCSLIB_LIBRARIES    ${WCSLIB_LIBRARY})
  endif()
endif()

# ---------------- Hard errors if missing ----------------
if(NOT CFITSIO_FOUND)
  message(FATAL_ERROR
    "CFITSIO not found.\n"
    "Install libcfitsio-dev (Ubuntu) or ensure fitsio.h + libcfitsio are in standard paths.\n"
    "If installed in a custom prefix, set CMAKE_PREFIX_PATH or install to /usr/local.\n"
  )
endif()

if(NOT WCSLIB_FOUND)
  message(FATAL_ERROR
    "WCSLIB not found.\n"
    "Install wcslib-dev/libwcslib-dev (Ubuntu) or ensure wcslib headers + libwcs are in standard paths.\n"
    "If installed in a custom prefix, set CMAKE_PREFIX_PATH or install to /usr/local.\n"
  )
endif()

# ---------------- Apply include/lib dirs and link ----------------
target_include_directories(ngps_acq PRIVATE
  ${CFITSIO_INCLUDE_DIRS}
  ${WCSLIB_INCLUDE_DIRS}
)

# Only add link directories if they exist (pkg-config path case)
if(DEFINED CFITSIO_LIBRARY_DIRS)
  target_link_directories(ngps_acq PRIVATE ${CFITSIO_LIBRARY_DIRS})
endif()
if(DEFINED WCSLIB_LIBRARY_DIRS)
  target_link_directories(ngps_acq PRIVATE ${WCSLIB_LIBRARY_DIRS})
endif()

target_link_libraries(ngps_acq PRIVATE
  ${CFITSIO_LIBRARIES}
  ${WCSLIB_LIBRARIES}
  m
)

# Nice debug printout
message(STATUS "CFITSIO include: ${CFITSIO_INCLUDE_DIRS}")
message(STATUS "CFITSIO libs:    ${CFITSIO_LIBRARIES}")
message(STATUS "WCSLIB include:  ${WCSLIB_INCLUDE_DIRS}")
message(STATUS "WCSLIB libs:     ${WCSLIB_LIBRARIES}")

