# -----------------------
# ExternalProject: torch
# -----------------------
set(proj_torch torch)

ExternalProject_Add(${proj_torch}
  PREFIX ${CMAKE_BINARY_DIR}/${proj_torch}
  URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip"
  DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/downloads
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

set(torch_DIR ${CMAKE_BINARY_DIR}/${proj_torch}/src/${proj_torch})

mark_as_superbuild(torch_DIR:PATH)