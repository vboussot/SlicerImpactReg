cmake_minimum_required(VERSION 3.5)

project(ImpactReg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/vboussot/SlicerImpactReg")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Valentin Boussot (University of Rennes, France), Cédric Hémon (University of Rennes, France), Jean-Louis Dillenseger (University of Rennes, France)")
set(EXTENSION_DESCRIPTION "")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/vboussot/SlicerImpactReg/main/ImpactReg.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ImpactReg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})


