
PROJECT(ABC)

FIND_PACKAGE(GenerateCLP REQUIRED)
IF(GenerateCLP_FOUND)
  INCLUDE(${GenerateCLP_USE_FILE})
ENDIF(GenerateCLP_FOUND)

LINK_DIRECTORIES(${ModuleDescriptionParser_BINARY_DIR})

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

IF (Slicer3_SOURCE_DIR)
  SET(LIBRARY_OUTPUT_PATH "${SLICER_LIBRARIES_DIR}/Plugins")
  SET(EXECUTABLE_OUTPUT_PATH "${SLICER_LIBRARIES_DIR}/Plugins")
ENDIF (Slicer3_SOURCE_DIR)

INCLUDE_DIRECTORIES( ${SlicerBaseCLI_BINARY_DIR} ${SlicerBaseCLI_SOURCE_DIR} )

###

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

###

SET(ABC_SRCS
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
  ../Engine/robust/KruskalMSTClusteringProcess.cxx
  ../Engine/spr/KMeansEstimator.cxx
  ../Engine/brainseg/EMSegmentationFilter_float+float.cxx
)

###

#ADD_LIBRARY(MuLib SHARED ${MU_SOURCE})
#SET_TARGET_PROPERTIES (MuLib PROPERTIES COMPILE_FLAGS "-Dmain=ModuleEntryPoint")
#TARGET_LINK_LIBRARIES (MuLib ${ITK_LIBRARIES})

###

SET (CLP ABC)

SET ( ${CLP}_SOURCE ${CLP}.cxx)
GENERATECLP(${CLP}_SOURCE ${CLP}.xml ${Slicer3_SOURCE_DIR}/Resources/NAMICLogo.h)
ADD_EXECUTABLE(${CLP} ${${CLP}_SOURCE} ${ABC_SRCS})
slicer3_set_plugins_output_path(${CLP})
TARGET_LINK_LIBRARIES (${CLP}
  ${ITK_LIBRARIES}
)

###

IF (Slicer3_SOURCE_DIR)
  # install each target in the production area (where it would appear in an
  # installation) and install each target in the developer area (for running
  # from a build)
  SET(TARGETS
    ABC
  )

  slicer3_install_plugins(${TARGETS})

  #INSTALL(TARGETS ${TARGETS}
  #  RUNTIME DESTINATION ${SLICER_INSTALL_LIBRARIES_DIR}/Plugins COMPONENT RuntimeLibraries
  #  LIBRARY DESTINATION ${SLICER_INSTALL_LIBRARIES_DIR}/Plugins COMPONENT RuntimeLibraries)

  #FOREACH(target ${TARGETS})
  #  INSTALL(TARGETS ${target}
  #    RUNTIME DESTINATION ${SLICER_INSTALL_LIBRARIES_DIR}/Plugins
  #    LIBRARY DESTINATION ${SLICER_INSTALL_LIBRARIES_DIR}/Plugins)
  #ENDFOREACH(target)

ENDIF (Slicer3_SOURCE_DIR)

