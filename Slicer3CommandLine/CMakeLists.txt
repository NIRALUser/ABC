
PROJECT(ABC)

cmake_minimum_required(VERSION 2.6)

# Find Slicer3. This will locate all the components needed to build modules 
# for Slicer3.
#
FIND_PACKAGE(Slicer3 QUIET)
IF (Slicer3_FOUND)
  INCLUDE(${Slicer3_USE_FILE})    
ELSE (Slicer3_FOUND)
  FIND_PACKAGE(ITK REQUIRED)
  IF (ITK_FOUND)
    INCLUDE(${USE_ITK_FILE})
  ELSE(ITK_FOUND)
    MESSAGE(FATAL_ERROR, "ITK not found. Please set ITK_DIR.")
  ENDIF (ITK_FOUND)

  FIND_PACKAGE(GenerateCLP REQUIRED)
  IF(GenerateCLP_FOUND)
    INCLUDE(${GenerateCLP_USE_FILE})
  ENDIF(GenerateCLP_FOUND)
ENDIF (Slicer3_FOUND)

###

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/basicimg
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/bias
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/brainseg
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/register
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/robust
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/spr
)

###

SET(ABC_SRCS
  ../Engine/common/Log.cxx
  ../Engine/common/MersenneTwisterRNG.cxx
  ../Engine/common/Timer.cxx
  ../Engine/common/muFile.cxx
  ../Engine/register/AmoebaOptimizer.cxx
  ../Engine/register/AtlasRegistrationMethod_float+float.cxx
  ../Engine/register/ChainedAffineTransform3D.cxx
  ../Engine/register/GradientDescentOptimizer.cxx
  ../Engine/register/PairRegistrationMethod_float.cxx
  ../Engine/register/PowellOptimizer.cxx
  ../Engine/register/SimulatedAnnealingOptimizer.cxx
  ../Engine/robust/FastMCDSampleFilter.cxx
  ../Engine/robust/KruskalMSTClusteringProcess.cxx
  ../Engine/spr/KMeansEstimator.cxx
  ../Engine/brainseg/EMSegmentationFilter_float+float.cxx
)

###

SET (CLP ABC)

SET ( ${CLP}_SOURCE ${CLP}.cxx)
GENERATECLP(${CLP}_SOURCE ${CLP}.xml ${CMAKE_CURRENT_SOURCE_DIR}/Resources/ITKLogo.h ${CMAKE_CURRENT_SOURCE_DIR}/Resources/NAMICLogo.h ${CMAKE_CURRENT_SOURCE_DIR}/Resources/SCILogo.h)
ADD_EXECUTABLE(${CLP} ${${CLP}_SOURCE} ${ABC_SRCS})
TARGET_LINK_LIBRARIES (${CLP}
  ${ITK_LIBRARIES}
)

#
# Install into a local plugins directory
#
IF (Slicer3_FOUND)
  set(TARGETS ABC)
  foreach(targ ${TARGETS})
    slicer3_set_plugins_output_path(${targ})
  endforeach(targ ${TARGETS})
  slicer3_install_plugins(${TARGETS})
ENDIF (Slicer3_FOUND)
