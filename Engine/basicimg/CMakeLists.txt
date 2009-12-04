
PROJECT(basicimg)

INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake)
IF (USE_ITK_FILE)
  INCLUDE(${USE_ITK_FILE})
ENDIF (USE_ITK_FILE)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../spr
)

ADD_EXECUTABLE(splitaxis splitaxis.cxx)
ADD_EXECUTABLE(testcrop testcrop.cxx)
ADD_EXECUTABLE(testdir testdir.cxx)
ADD_EXECUTABLE(testit testit.cxx)
ADD_EXECUTABLE(testquant
  ../common/MersenneTwisterRNG.cxx
  ../spr/KMeansEstimator.cxx
  testquant.cxx
)

TARGET_LINK_LIBRARIES(splitaxis ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(testcrop ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(testdir ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(testit ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(testquant ${ITK_LIBRARIES})
