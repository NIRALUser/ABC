
CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

PROJECT(spr)

FIND_PACKAGE(ITK)
IF (ITK_FOUND)
  INCLUDE(${USE_ITK_FILE})
ELSE (ITK_FOUND)
  MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
ENDIF (ITK_FOUND)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../qhull
  ${CMAKE_CURRENT_SOURCE_DIR}/../robust
) 

ADD_EXECUTABLE(spr
  ../common/Log.cxx
  ../common/MersenneTwisterRNG.cxx
  ../qhull/user.c
  ../qhull/global.c
  ../qhull/stat.c
  ../qhull/io.c
  ../qhull/geom2.c
  ../qhull/poly2.c
  ../qhull/merge.c
  ../qhull/qhull.c
  ../qhull/geom.c
  ../qhull/poly.c
  ../qhull/qset.c
  ../qhull/mem.c
  ../robust/FastMCDSampleFilter.cxx
  ../robust/QHullMSTClusteringProcess.cxx
  ../robust/PrimMSTClusteringProcess.cxx
  DiagonalKernelDensityEstimator.cxx
  KMeansEstimator.cxx
  KNNClassifier.cxx
  MultivariateGaussian.cxx
  ReducedSetDensityEstimator.cxx
  SphericalKernelDensityEstimator.cxx
  TreeStructuredVectorQuantizer.cxx
  test.cxx
)

ADD_EXECUTABLE(testbw
  ../common/Log.cxx
  ../common/MersenneTwisterRNG.cxx
  ../qhull/user.c
  ../qhull/global.c
  ../qhull/stat.c
  ../qhull/io.c
  ../qhull/geom2.c
  ../qhull/poly2.c
  ../qhull/merge.c
  ../qhull/qhull.c
  ../qhull/geom.c
  ../qhull/poly.c
  ../qhull/qset.c
  ../qhull/mem.c
  ../robust/QHullMSTClusteringProcess.cxx
  SphericalKernelDensityEstimator.cxx
  DiagonalKernelDensityEstimator.cxx
  KMeansEstimator.cxx
  testbw.cxx
)

TARGET_LINK_LIBRARIES(spr ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(testbw ${ITK_LIBRARIES})
