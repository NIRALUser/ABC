CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

PROJECT(ABC_CLI)

find_package( ITK REQUIRED )
IF (ITK_USE_FILE)
  INCLUDE(${ITK_USE_FILE})
ENDIF (ITK_USE_FILE)

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
OPTION( ABC_CLI_NO_SUFFIX "Name ABC_CLI ABC (no suffix)" OFF)
if( ABC_CLI_NO_SUFFIX AND COMPILE_SLICER4COMMANDLINE )
  message(WARNING "Setting ABC_CLI_NO_SUFFIX and COMPILE_SLICER4COMMANDLINE to ON will create errors as multiple targets will have the same name.")
endif()
if( ABC_CLI_NO_SUFFIX )
  set( TargetName ABC )
else()
  set( TargetName ABC_CLI )
endif()
ADD_EXECUTABLE(${TargetName} ${ABC_CLI_SRCS})
TARGET_LINK_LIBRARIES(${TargetName} ${ITK_LIBRARIES})

install(TARGETS ${TargetName} RUNTIME DESTINATION bin)

