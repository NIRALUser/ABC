
PROJECT(robust)

INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake)
IF (USE_ITK_FILE)
  INCLUDE(${USE_ITK_FILE})
ENDIF (USE_ITK_FILE)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../qhull
) 

ADD_EXECUTABLE(test
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
  WeightedFastMCDSampleFilter.cxx
  KruskalMSTClusteringProcess.cxx
  QHullMSTClusteringProcess.cxx
  test.cxx
)

TARGET_LINK_LIBRARIES(test ${ITK_LIBRARIES})
