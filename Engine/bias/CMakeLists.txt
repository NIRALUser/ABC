
PROJECT(bias)

INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake)
IF (USE_ITK_FILE)
  INCLUDE(${USE_ITK_FILE})
ENDIF (USE_ITK_FILE)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../register
)

ADD_EXECUTABLE(test
  test.cxx
)

ADD_EXECUTABLE(bias
  bias.cxx
)

TARGET_LINK_LIBRARIES(bias ${ITK_LIBRARIES})
