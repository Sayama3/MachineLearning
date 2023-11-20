cmake_minimum_required(VERSION 3.24)

add_subdirectory(lib)

project(MachineLearning VERSION 0.0.1 DESCRIPTION "The core library of the MachineLearning library.")
set(LIBRARY_TARGET_NAME MLCore)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_compile_definitions(MACHINE_LEARNING_EXPORTS)


option(ML_SHARED_BUILD "Use the MachineLearning library in shared mode." ON)
option(ML_USE_LOGS "Enable the logs in the MachineLearning library." ON)


set(ML_SRC_FILES
        src/library.cpp
        include/library.h
)

if(ML_SHARED_BUILD)
    add_library(${LIBRARY_TARGET_NAME} SHARED
        ${ML_SRC_FILES}
    )
else()
    add_library(${LIBRARY_TARGET_NAME}
        ${ML_SRC_FILES}
    )
endif ()

if(CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]")
    add_compile_definitions(ML_DEBUG)
endif()

set_property(TARGET ${LIBRARY_TARGET_NAME} PROPERTY
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)

include(GenerateExportHeader)
generate_export_header(${LIBRARY_TARGET_NAME}
    BASE_NAME "MachineLearning"
    EXPORT_MACRO_NAME "ML_API"
    EXPORT_FILE_NAME "export/machinelearning_export.hpp"
)

target_include_directories(${LIBRARY_TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/export)

add_library(${PROJECT_NAME}::${LIBRARY_TARGET_NAME} ALIAS ${LIBRARY_TARGET_NAME})

set_target_properties(${LIBRARY_TARGET_NAME} PROPERTIES
    VERSION       ${${PROJECT_NAME}_VERSION}
    PUBLIC_HEADER "include/MachineLearning/MachineLearning.hpp"
    PUBLIC cxx_std_20
)

target_include_directories(${LIBRARY_TARGET_NAME} PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_include_directories(${LIBRARY_TARGET_NAME} PRIVATE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
)


target_link_libraries(${LIBRARY_TARGET_NAME} PUBLIC glm)
if(ML_USE_LOGS)
    #TODO: See if we really need the spdlog library.
    target_link_libraries(${LIBRARY_TARGET_NAME} PRIVATE spdlog::spdlog_header_only)
    target_compile_definitions(${LIBRARY_TARGET_NAME} PRIVATE ML_USE_LOGS)
endif()

#target_precompile_headers(${LIBRARY_TARGET_NAME}
#        PUBLIC
#        <array>
#        <cstdint>
#        <filesystem>
#        <fstream>
#        <functional>
#        <iostream>
#        <memory>
#        <stdexcept>
#        <string>
#        <sstream>
#        <stdexcept>
#        <unordered_map>
#        <utility>
#        <vector>
#        <glm/glm.hpp>
#        <glm/gtc/quaternion.hpp>
#        <glm/gtc/matrix_transform.hpp>
#        <glm/gtc/type_ptr.hpp>
#        <glm/gtx/quaternion.hpp>
#        <glm/gtx/string_cast.hpp>
#        <machinelearning_export.hpp>
#        )

# Specify installation targets, typology and destination folders.
install(TARGETS ${LIBRARY_TARGET_NAME}
EXPORT  ${PROJECT_NAME}
LIBRARY       DESTINATION "${CMAKE_INSTALL_LIBDIR}"                            COMPONENT shlib
ARCHIVE       DESTINATION "${CMAKE_INSTALL_LIBDIR}"                            COMPONENT lib
RUNTIME       DESTINATION "${CMAKE_INSTALL_BINDIR}"                            COMPONENT bin
PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${LIBRARY_TARGET_NAME}" COMPONENT dev)


message(STATUS "Created target ${LIBRARY_TARGET_NAME} for export ${PROJECT_NAME}.")
