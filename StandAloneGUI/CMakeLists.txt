
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(ABC_GUI)

FIND_PACKAGE(ITK)
IF (ITK_FOUND)
  INCLUDE(${USE_ITK_FILE})
ELSE (ITK_FOUND)
  MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
ENDIF (ITK_FOUND)

FIND_PACKAGE(FLTK REQUIRED NO_MODULE)
INCLUDE(${FLTK_USE_FILE})

FIND_PACKAGE(OpenGL REQUIRED)

INCLUDE_DIRECTORIES(
  ${FLTK_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/basicimg
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/bias
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/brainseg
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/register
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/robust
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/spr
  ${CMAKE_CURRENT_SOURCE_DIR}/../Engine/xmlio
)

SET(ABC_GUI_SRCS
  ../Engine/brainseg/EMSegmentationFilter_float+float.cxx
  ../Engine/brainseg/EMSParameters.cxx
  ../Engine/brainseg/filterFloatImages.cxx
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
  GUI.cxx
  main_gui.cxx
)

ADD_EXECUTABLE(ABC_GUI ${ABC_GUI_SRCS})

#TARGET_LINK_LIBRARIES(ABC_GUI ${FLTK_LIBRARIES} ${OPENGL_LIBRARIES} ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(ABC_GUI ${ITK_LIBRARIES} fltk)

