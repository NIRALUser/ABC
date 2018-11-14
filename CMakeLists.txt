project(ABC)
cmake_minimum_required(VERSION 3.0.2)

option(ABC_SUPERBUILD "Download and build the proper version of ITK" ON )
OPTION(COMPILE_SLICER4COMMANDLINE "Compile Slicer4CommandLine" ON)
OPTION(COMPILE_BRAINSEG "Compile brainseg" OFF)
OPTION(COMPILE_STANDALONEGUI "Compile StandAloneGUI" OFF)
OPTION(COMPILE_COMMANDLINE "Compile CommandLine" OFF)

if( ${ABC_SUPERBUILD} )
	include(${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild.cmake)
else()
	include(${CMAKE_CURRENT_SOURCE_DIR}/ABC.cmake)
endif()