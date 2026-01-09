cmake_minimum_required(VERSION 3.5)

project(ImpactReg)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/vboussot/SlicerImpactReg")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Valentin Boussot (University of Rennes, France), Cédric Hémon (University of Rennes, France), Jean-Louis Dillenseger (University of Rennes, France)")
set(EXTENSION_DESCRIPTION "Slicer IMPACT-Reg is a 3D Slicer module dedicated to multimodal medical image registration. It integrates the IMPACT deep semantic similarity metric within the Elastix registration engine and exposes predefined registration presets through a simple graphical interface.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/vboussot/SlicerImpactReg/main/ImpactReg.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/vboussot/SlicerImpactReg/main/docs/Registration.png https://raw.githubusercontent.com/vboussot/SlicerImpactReg/main/docs/Evaluation.png")
set(EXTENSION_DEPENDS "KonfAI")

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


