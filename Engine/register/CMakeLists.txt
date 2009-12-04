
PROJECT(register)

INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake)
IF (USE_ITK_FILE)
  INCLUDE(${USE_ITK_FILE})
ENDIF (USE_ITK_FILE)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/../basicimg
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${CMAKE_CURRENT_SOURCE_DIR}/../spr
)

ADD_EXECUTABLE(mireg_affine
  ../common/muFile.cxx
  ../common/Log.cxx
  ../common/Timer.cxx
  ../common/MersenneTwisterRNG.cxx
  ../spr/KMeansEstimator.cxx
  AmoebaOptimizer.cxx
  ChainedAffineTransform3D.cxx
  GradientDescentOptimizer.cxx
  PowellOptimizer.cxx
  SimulatedAnnealingOptimizer.cxx
  mireg_affine.cxx
)

ADD_EXECUTABLE(mireg_bspline
  ../common/muFile.cxx
  ../common/Log.cxx
  ../common/Timer.cxx
  ../common/MersenneTwisterRNG.cxx
  ../spr/KMeansEstimator.cxx
  AmoebaOptimizer.cxx
  ChainedAffineTransform3D.cxx
  GradientDescentOptimizer.cxx
  PowellOptimizer.cxx
  SimulatedAnnealingOptimizer.cxx
  mireg_bspline.cxx
)

ADD_EXECUTABLE(resample_label
  ../common/Log.cxx
  ChainedAffineTransform3D.cxx
  resample_label.cxx
)

ADD_EXECUTABLE(resample_fb
  ../common/Log.cxx
  ChainedAffineTransform3D.cxx
  resample_fb.cxx
)

ADD_EXECUTABLE(chaff2itk
  ../common/Log.cxx
  ChainedAffineTransform3D.cxx
  chaff2itk.cxx
)

ADD_EXECUTABLE(test_chaff
  ChainedAffineTransform3D.cxx
  test_chaff.cxx
)

TARGET_LINK_LIBRARIES(mireg_affine ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(mireg_bspline ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(resample_label ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(resample_fb ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(chaff2itk ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(test_chaff ${ITK_LIBRARIES})
