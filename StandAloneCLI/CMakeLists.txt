CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

PROJECT(ABC_CLI)

INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake)
IF (USE_ITK_FILE)
  INCLUDE(${USE_ITK_FILE})
ENDIF (USE_ITK_FILE)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/basicimg
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/bias
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/brainseg
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/register
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/robust
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/spr
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/xmlio
)

SET(ABC_CLI_SRCS
  ../Engine/brainseg/EMSParameters.cxx
  ../Engine/brainseg/EMSegmentationFilter_float+float.cxx
  ../Engine/brainseg/runEMS.cxx
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
  ../Engine/xmlio/EMSParametersXMLFile.cxx
  main_cli.cxx
)

ADD_EXECUTABLE(ABC_CLI ${ABC_CLI_SRCS})

TARGET_LINK_LIBRARIES(ABC_CLI ${ITK_LIBRARIES})
