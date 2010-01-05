CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

PROJECT(BRAINSEG)

INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake)
IF (USE_ITK_FILE)
  INCLUDE(${USE_ITK_FILE})
ENDIF (USE_ITK_FILE)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../aniso
  ${CMAKE_CURRENT_SOURCE_DIR}/../basicimg
  ${CMAKE_CURRENT_SOURCE_DIR}/../bias
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../conn
  ${CMAKE_CURRENT_SOURCE_DIR}/../register
  ${CMAKE_CURRENT_SOURCE_DIR}/../robust
  ${CMAKE_CURRENT_SOURCE_DIR}/../spr
  ${CMAKE_CURRENT_SOURCE_DIR}/../xmlio
)

SET(BRAINSEG_SRCS
  ../common/Log.cxx
  ../common/MersenneTwisterRNG.cxx
  ../common/Timer.cxx
  ../common/muFile.cxx
  ../register/AmoebaOptimizer.cxx
  ../register/AtlasRegistrationMethod_float+float.cxx
  ../register/ChainedAffineTransform3D.cxx
  ../register/GradientDescentOptimizer.cxx
  ../register/PairRegistrationMethod_float.cxx
  ../register/PowellOptimizer.cxx
  ../register/SimulatedAnnealingOptimizer.cxx
  ../robust/KruskalMSTClusteringProcess.cxx
  ../spr/KMeansEstimator.cxx
  ../xmlio/EMSParametersXMLFile.cxx
  EMSParameters.cxx
  EMSegmentationFilter_float+float.cxx
  main.cxx
  runEMS.cxx
)

ADD_EXECUTABLE(brainseg ${BRAINSEG_SRCS})

TARGET_LINK_LIBRARIES(brainseg ${ITK_LIBRARIES})
