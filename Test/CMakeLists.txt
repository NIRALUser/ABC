CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

PROJECT(ABCTEST)

INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake)
IF (USE_ITK_FILE)
  INCLUDE(${USE_ITK_FILE})
ENDIF (USE_ITK_FILE)

ENABLE_TESTING()

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/aniso
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/basicimg
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/bias
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/brainseg
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/conn
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/register
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/robust
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/spr
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/xmlio
)

ADD_EXECUTABLE(gentest
  gentest.cxx
  ../Engine/register/ChainedAffineTransform3D.cxx
)

ADD_EXECUTABLE(ABCTestAll
  ABCTestAll.cxx
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
  ../Engine/brainseg/EMSegmentationFilter_float+float.cxx
  ../Engine/brainseg/EMSParameters.cxx
  ../Engine/brainseg/runEMS.cxx
)

TARGET_LINK_LIBRARIES(gentest ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(ABCTestAll ${ITK_LIBRARIES})

ADD_TEST(ABCTestAll ${EXECUTABLE_OUTPUT_PATH}/ABCTestAll ${CMAKE_CURRENT_SOURCE_DIR}/Atlas ${CMAKE_CURRENT_SOURCE_DIR}/Data ABCTestAll-out)
