include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
set(PRIMARY_PROJECT_NAME ABC)

option(USE_SYSTEM_ITK "Build using an externally defined version of ITK" OFF)

set(proj ITK)

if(NOT ( DEFINED "USE_SYSTEM_${proj}" AND "${USE_SYSTEM_${proj}}" ) )
  # Set CMake OSX variable to pass down the external project
  set(CMAKE_OSX_EXTERNAL_PROJECT_ARGS)
  if(APPLE)
    list(APPEND CMAKE_OSX_EXTERNAL_PROJECT_ARGS
      -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
      -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
      -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
  endif()
  set(${proj}_INSTALL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${proj}-install")
  set(${proj}_CMAKE_OPTIONS
      -DBUILD_TESTING:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
      -DCMAKE_INSTALL_PREFIX:PATH=${${proj}_INSTALL_PATH}
      -DITK_LEGACY_REMOVE:BOOL=OFF
      -DITK_FUTURE_LEGACY_REMOVE:BOOL=OFF
      -DITKV3_COMPATIBILITY:BOOL=OFF
      -DITK_USE_REVIEW:BOOL=ON
      -DITK_WRAP_PYTHON:BOOL=OFF
      -DModule_ITKReview:BOOL=ON
      -DModule_ITKIODCMTK:BOOL=ON
      -DModule_MGHIO:BOOL=ON
      -DITK_BUILD_DEFAULT_MODULES:BOOL=ON
      -DITK_WRAPPING:BOOL=OFF #${BUILD_SHARED_LIBS} ## HACK:  QUICK CHANGE
    )
  ### --- End Project specific additions
  set(${proj}_REPOSITORY http://itk.org/ITK.git)
  set(${proj}_GIT_TAG "v4.8.2")
  set(ITK_VERSION_ID ITK-4.8)

  ExternalProject_Add(${proj}
    GIT_REPOSITORY ${${proj}_REPOSITORY}
    GIT_TAG ${${proj}_GIT_TAG}
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}
    BINARY_DIR ${proj}-build
    LOG_CONFIGURE 0  # Wrap configure in script to ignore log output from dashboards
    LOG_BUILD     0  # Wrap build in script to to ignore log output from dashboards
    LOG_TEST      0  # Wrap test in script to to ignore log output from dashboards
    LOG_INSTALL   0  # Wrap install in script to to ignore log output from dashboards
    ${cmakeversion_external_update} "${cmakeversion_external_update_value}"
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
      ${CMAKE_OSX_EXTERNAL_PROJECT_ARGS}
      ${COMMON_EXTERNAL_PROJECT_ARGS}
      ${${proj}_CMAKE_OPTIONS}
  )
  set(${proj}_DIR ${CMAKE_BINARY_DIR}/${proj}-install/lib/cmake/${ITK_VERSION_ID})
else()
  # if(${USE_SYSTEM_${proj}})  
    find_package(${proj} ${ITK_VERSION_MAJOR} REQUIRED)
    include(${ITK_USE_FILE})
    message("USING the system ${proj}, set ${proj}_DIR=${${proj}_DIR}")
  # endif()
  if( NOT TARGET ${proj} )
    # The project is provided using ${extProjName}_DIR, nevertheless since other
    # project may depend on ${extProjName}, let's add an 'empty' one
    ExternalProject_Add(${proj}
    SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj}
    BINARY_DIR ${proj}-build
    DOWNLOAD_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""    
    INSTALL_COMMAND ""
    )
  endif()

endif()

set(proj ${PRIMARY_PROJECT_NAME}-inner)
set(${proj}_DEPENDENCIES ITK)
set(${proj}_INSTALL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${proj}-install")
set(INSTALL_RUNTIME_DESTINATION bin)

ExternalProject_Add(${proj}
  DOWNLOAD_COMMAND ""  
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${proj}-build
  LOG_CONFIGURE 0  # Wrap configure in script to ignore log output from dashboards
  LOG_BUILD     0  # Wrap build in script to to ignore log output from dashboards
  LOG_TEST      0  # Wrap test in script to to ignore log output from dashboards
  LOG_INSTALL   0  # Wrap install in script to to ignore log output from dashboards
  ${cmakeversion_external_update} "${cmakeversion_external_update_value}"
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=${${proj}_INSTALL_PATH}
    -DINSTALL_RUNTIME_DESTINATION:PATH=${INSTALL_RUNTIME_DESTINATION}
    -DABC_SUPERBUILD:BOOL=OFF
    -DITK_DIR:PATH=${ITK_DIR}
    -DCOMPILE_SLICER4COMMANDLINE:BOOL=${COMPILE_SLICER4COMMANDLINE}
    -DCOMPILE_BRAINSEG:BOOL=${COMPILE_BRAINSEG}
    -DCOMPILE_STANDALONEGUI:BOOL=${COMPILE_STANDALONEGUI}
    -DCOMPILE_COMMANDLINE:BOOL=${COMPILE_COMMANDLINE}
    ${CMAKE_OSX_EXTERNAL_PROJECT_ARGS}
  DEPENDS
    ${${proj}_DEPENDENCIES}
)
