########## MACROS ###########################################################################
#############################################################################################

function(conan_message MESSAGE_OUTPUT)
    if(NOT CONAN_CMAKE_SILENT_OUTPUT)
        message(${ARGV${0}})
    endif()
endfunction()


macro(conan_find_apple_frameworks FRAMEWORKS_FOUND FRAMEWORKS FRAMEWORKS_DIRS)
    if(APPLE)
        foreach(_FRAMEWORK ${FRAMEWORKS})
            # https://cmake.org/pipermail/cmake-developers/2017-August/030199.html
            find_library(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND NAMES ${_FRAMEWORK} PATHS ${FRAMEWORKS_DIRS} CMAKE_FIND_ROOT_PATH_BOTH)
            if(CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND)
                list(APPEND ${FRAMEWORKS_FOUND} ${CONAN_FRAMEWORK_${_FRAMEWORK}_FOUND})
            else()
                message(FATAL_ERROR "Framework library ${_FRAMEWORK} not found in paths: ${FRAMEWORKS_DIRS}")
            endif()
        endforeach()
    endif()
endmacro()


function(conan_package_library_targets libraries package_libdir deps out_libraries out_libraries_target build_type package_name)
    unset(_CONAN_ACTUAL_TARGETS CACHE)
    unset(_CONAN_FOUND_SYSTEM_LIBS CACHE)
    foreach(_LIBRARY_NAME ${libraries})
        find_library(CONAN_FOUND_LIBRARY NAMES ${_LIBRARY_NAME} PATHS ${package_libdir}
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(CONAN_FOUND_LIBRARY)
            conan_message(STATUS "Library ${_LIBRARY_NAME} found ${CONAN_FOUND_LIBRARY}")
            list(APPEND _out_libraries ${CONAN_FOUND_LIBRARY})
            if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
                # Create a micro-target for each lib/a found
                string(REGEX REPLACE "[^A-Za-z0-9.+_-]" "_" _LIBRARY_NAME ${_LIBRARY_NAME})
                set(_LIB_NAME CONAN_LIB::${package_name}_${_LIBRARY_NAME}${build_type})
                if(NOT TARGET ${_LIB_NAME})
                    # Create a micro-target for each lib/a found
                    add_library(${_LIB_NAME} UNKNOWN IMPORTED)
                    set_target_properties(${_LIB_NAME} PROPERTIES IMPORTED_LOCATION ${CONAN_FOUND_LIBRARY})
                    set(_CONAN_ACTUAL_TARGETS ${_CONAN_ACTUAL_TARGETS} ${_LIB_NAME})
                else()
                    conan_message(STATUS "Skipping already existing target: ${_LIB_NAME}")
                endif()
                list(APPEND _out_libraries_target ${_LIB_NAME})
            endif()
            conan_message(STATUS "Found: ${CONAN_FOUND_LIBRARY}")
        else()
            conan_message(STATUS "Library ${_LIBRARY_NAME} not found in package, might be system one")
            list(APPEND _out_libraries_target ${_LIBRARY_NAME})
            list(APPEND _out_libraries ${_LIBRARY_NAME})
            set(_CONAN_FOUND_SYSTEM_LIBS "${_CONAN_FOUND_SYSTEM_LIBS};${_LIBRARY_NAME}")
        endif()
        unset(CONAN_FOUND_LIBRARY CACHE)
    endforeach()

    if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
        # Add all dependencies to all targets
        string(REPLACE " " ";" deps_list "${deps}")
        foreach(_CONAN_ACTUAL_TARGET ${_CONAN_ACTUAL_TARGETS})
            set_property(TARGET ${_CONAN_ACTUAL_TARGET} PROPERTY INTERFACE_LINK_LIBRARIES "${_CONAN_FOUND_SYSTEM_LIBS};${deps_list}")
        endforeach()
    endif()

    set(${out_libraries} ${_out_libraries} PARENT_SCOPE)
    set(${out_libraries_target} ${_out_libraries_target} PARENT_SCOPE)
endfunction()


########### FOUND PACKAGE ###################################################################
#############################################################################################

include(FindPackageHandleStandardArgs)

conan_message(STATUS "Conan: Using autogenerated FindPCRE.cmake")
set(PCRE_FOUND 1)
set(PCRE_VERSION "8.45")

find_package_handle_standard_args(PCRE REQUIRED_VARS
                                  PCRE_VERSION VERSION_VAR PCRE_VERSION)
mark_as_advanced(PCRE_FOUND PCRE_VERSION)

set(PCRE_COMPONENTS PCRE::libpcreposix PCRE::libpcre PCRE::libpcre16 PCRE::libpcre32)

if(PCRE_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${PCRE_FIND_COMPONENTS})
        list(FIND PCRE_COMPONENTS "PCRE::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'PCRE'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'PCRE'")
        endif()
    endforeach()
endif()

########### VARIABLES #######################################################################
#############################################################################################


set(PCRE_INCLUDE_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_INCLUDE_DIR "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_INCLUDES "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_RES_DIRS )
set(PCRE_DEFINITIONS "-DPCRE_STATIC=1")
set(PCRE_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(PCRE_COMPILE_DEFINITIONS "PCRE_STATIC=1")
set(PCRE_COMPILE_OPTIONS_LIST "" "")
set(PCRE_COMPILE_OPTIONS_C "")
set(PCRE_COMPILE_OPTIONS_CXX "")
set(PCRE_LIBRARIES_TARGETS "") # Will be filled later, if CMake 3
set(PCRE_LIBRARIES "") # Will be filled later
set(PCRE_LIBS "") # Same as PCRE_LIBRARIES
set(PCRE_SYSTEM_LIBS )
set(PCRE_FRAMEWORK_DIRS )
set(PCRE_FRAMEWORKS )
set(PCRE_FRAMEWORKS_FOUND "") # Will be filled later
set(PCRE_BUILD_MODULES_PATHS )

conan_find_apple_frameworks(PCRE_FRAMEWORKS_FOUND "${PCRE_FRAMEWORKS}" "${PCRE_FRAMEWORK_DIRS}")

mark_as_advanced(PCRE_INCLUDE_DIRS
                 PCRE_INCLUDE_DIR
                 PCRE_INCLUDES
                 PCRE_DEFINITIONS
                 PCRE_LINKER_FLAGS_LIST
                 PCRE_COMPILE_DEFINITIONS
                 PCRE_COMPILE_OPTIONS_LIST
                 PCRE_LIBRARIES
                 PCRE_LIBS
                 PCRE_LIBRARIES_TARGETS)

# Find the real .lib/.a and add them to PCRE_LIBS and PCRE_LIBRARY_LIST
set(PCRE_LIBRARY_LIST pcreposix pcre pcre16 pcre32)
set(PCRE_LIB_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_PCRE_DEPENDENCIES "${PCRE_FRAMEWORKS_FOUND} ${PCRE_SYSTEM_LIBS} BZip2::BZip2;ZLIB::ZLIB")

conan_package_library_targets("${PCRE_LIBRARY_LIST}"  # libraries
                              "${PCRE_LIB_DIRS}"      # package_libdir
                              "${_PCRE_DEPENDENCIES}"  # deps
                              PCRE_LIBRARIES            # out_libraries
                              PCRE_LIBRARIES_TARGETS    # out_libraries_targets
                              ""                          # build_type
                              "PCRE")                                      # package_name

set(PCRE_LIBS ${PCRE_LIBRARIES})

foreach(_FRAMEWORK ${PCRE_FRAMEWORKS_FOUND})
    list(APPEND PCRE_LIBRARIES_TARGETS ${_FRAMEWORK})
    list(APPEND PCRE_LIBRARIES ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${PCRE_SYSTEM_LIBS})
    list(APPEND PCRE_LIBRARIES_TARGETS ${_SYSTEM_LIB})
    list(APPEND PCRE_LIBRARIES ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(PCRE_LIBRARIES_TARGETS "${PCRE_LIBRARIES_TARGETS};BZip2::BZip2;ZLIB::ZLIB")
set(PCRE_LIBRARIES "${PCRE_LIBRARIES};BZip2::BZip2;ZLIB::ZLIB")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})


########### COMPONENT libpcre32 VARIABLES #############################################

set(PCRE_libpcre32_INCLUDE_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre32_INCLUDE_DIR "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre32_INCLUDES "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre32_LIB_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/lib")
set(PCRE_libpcre32_RES_DIRS )
set(PCRE_libpcre32_DEFINITIONS "-DPCRE_STATIC=1")
set(PCRE_libpcre32_COMPILE_DEFINITIONS "PCRE_STATIC=1")
set(PCRE_libpcre32_COMPILE_OPTIONS_C "")
set(PCRE_libpcre32_COMPILE_OPTIONS_CXX "")
set(PCRE_libpcre32_LIBS pcre32)
set(PCRE_libpcre32_SYSTEM_LIBS )
set(PCRE_libpcre32_FRAMEWORK_DIRS )
set(PCRE_libpcre32_FRAMEWORKS )
set(PCRE_libpcre32_BUILD_MODULES_PATHS )
set(PCRE_libpcre32_DEPENDENCIES )
set(PCRE_libpcre32_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)

########### COMPONENT libpcre16 VARIABLES #############################################

set(PCRE_libpcre16_INCLUDE_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre16_INCLUDE_DIR "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre16_INCLUDES "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre16_LIB_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/lib")
set(PCRE_libpcre16_RES_DIRS )
set(PCRE_libpcre16_DEFINITIONS "-DPCRE_STATIC=1")
set(PCRE_libpcre16_COMPILE_DEFINITIONS "PCRE_STATIC=1")
set(PCRE_libpcre16_COMPILE_OPTIONS_C "")
set(PCRE_libpcre16_COMPILE_OPTIONS_CXX "")
set(PCRE_libpcre16_LIBS pcre16)
set(PCRE_libpcre16_SYSTEM_LIBS )
set(PCRE_libpcre16_FRAMEWORK_DIRS )
set(PCRE_libpcre16_FRAMEWORKS )
set(PCRE_libpcre16_BUILD_MODULES_PATHS )
set(PCRE_libpcre16_DEPENDENCIES )
set(PCRE_libpcre16_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)

########### COMPONENT libpcre VARIABLES #############################################

set(PCRE_libpcre_INCLUDE_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre_INCLUDE_DIR "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre_INCLUDES "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcre_LIB_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/lib")
set(PCRE_libpcre_RES_DIRS )
set(PCRE_libpcre_DEFINITIONS "-DPCRE_STATIC=1")
set(PCRE_libpcre_COMPILE_DEFINITIONS "PCRE_STATIC=1")
set(PCRE_libpcre_COMPILE_OPTIONS_C "")
set(PCRE_libpcre_COMPILE_OPTIONS_CXX "")
set(PCRE_libpcre_LIBS pcre)
set(PCRE_libpcre_SYSTEM_LIBS )
set(PCRE_libpcre_FRAMEWORK_DIRS )
set(PCRE_libpcre_FRAMEWORKS )
set(PCRE_libpcre_BUILD_MODULES_PATHS )
set(PCRE_libpcre_DEPENDENCIES BZip2::BZip2 ZLIB::ZLIB)
set(PCRE_libpcre_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)

########### COMPONENT libpcreposix VARIABLES #############################################

set(PCRE_libpcreposix_INCLUDE_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcreposix_INCLUDE_DIR "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcreposix_INCLUDES "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/include")
set(PCRE_libpcreposix_LIB_DIRS "/home/blackbird/.conan/data/pcre/8.45/_/_/package/87087120c448298530c012e627c1a0b8f062586d/lib")
set(PCRE_libpcreposix_RES_DIRS )
set(PCRE_libpcreposix_DEFINITIONS )
set(PCRE_libpcreposix_COMPILE_DEFINITIONS )
set(PCRE_libpcreposix_COMPILE_OPTIONS_C "")
set(PCRE_libpcreposix_COMPILE_OPTIONS_CXX "")
set(PCRE_libpcreposix_LIBS pcreposix)
set(PCRE_libpcreposix_SYSTEM_LIBS )
set(PCRE_libpcreposix_FRAMEWORK_DIRS )
set(PCRE_libpcreposix_FRAMEWORKS )
set(PCRE_libpcreposix_BUILD_MODULES_PATHS )
set(PCRE_libpcreposix_DEPENDENCIES PCRE::libpcre)
set(PCRE_libpcreposix_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)


########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)

if(NOT BZip2_FOUND)
    find_dependency(BZip2 REQUIRED)
else()
    conan_message(STATUS "Conan: Dependency BZip2 already found")
endif()

if(NOT ZLIB_FOUND)
    find_dependency(ZLIB REQUIRED)
else()
    conan_message(STATUS "Conan: Dependency ZLIB already found")
endif()


########## FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #######################################
#############################################################################################

########## COMPONENT libpcre32 FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(PCRE_libpcre32_FRAMEWORKS_FOUND "")
conan_find_apple_frameworks(PCRE_libpcre32_FRAMEWORKS_FOUND "${PCRE_libpcre32_FRAMEWORKS}" "${PCRE_libpcre32_FRAMEWORK_DIRS}")

set(PCRE_libpcre32_LIB_TARGETS "")
set(PCRE_libpcre32_NOT_USED "")
set(PCRE_libpcre32_LIBS_FRAMEWORKS_DEPS ${PCRE_libpcre32_FRAMEWORKS_FOUND} ${PCRE_libpcre32_SYSTEM_LIBS} ${PCRE_libpcre32_DEPENDENCIES})
conan_package_library_targets("${PCRE_libpcre32_LIBS}"
                              "${PCRE_libpcre32_LIB_DIRS}"
                              "${PCRE_libpcre32_LIBS_FRAMEWORKS_DEPS}"
                              PCRE_libpcre32_NOT_USED
                              PCRE_libpcre32_LIB_TARGETS
                              ""
                              "PCRE_libpcre32")

set(PCRE_libpcre32_LINK_LIBS ${PCRE_libpcre32_LIB_TARGETS} ${PCRE_libpcre32_LIBS_FRAMEWORKS_DEPS})

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

########## COMPONENT libpcre16 FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(PCRE_libpcre16_FRAMEWORKS_FOUND "")
conan_find_apple_frameworks(PCRE_libpcre16_FRAMEWORKS_FOUND "${PCRE_libpcre16_FRAMEWORKS}" "${PCRE_libpcre16_FRAMEWORK_DIRS}")

set(PCRE_libpcre16_LIB_TARGETS "")
set(PCRE_libpcre16_NOT_USED "")
set(PCRE_libpcre16_LIBS_FRAMEWORKS_DEPS ${PCRE_libpcre16_FRAMEWORKS_FOUND} ${PCRE_libpcre16_SYSTEM_LIBS} ${PCRE_libpcre16_DEPENDENCIES})
conan_package_library_targets("${PCRE_libpcre16_LIBS}"
                              "${PCRE_libpcre16_LIB_DIRS}"
                              "${PCRE_libpcre16_LIBS_FRAMEWORKS_DEPS}"
                              PCRE_libpcre16_NOT_USED
                              PCRE_libpcre16_LIB_TARGETS
                              ""
                              "PCRE_libpcre16")

set(PCRE_libpcre16_LINK_LIBS ${PCRE_libpcre16_LIB_TARGETS} ${PCRE_libpcre16_LIBS_FRAMEWORKS_DEPS})

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

########## COMPONENT libpcre FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(PCRE_libpcre_FRAMEWORKS_FOUND "")
conan_find_apple_frameworks(PCRE_libpcre_FRAMEWORKS_FOUND "${PCRE_libpcre_FRAMEWORKS}" "${PCRE_libpcre_FRAMEWORK_DIRS}")

set(PCRE_libpcre_LIB_TARGETS "")
set(PCRE_libpcre_NOT_USED "")
set(PCRE_libpcre_LIBS_FRAMEWORKS_DEPS ${PCRE_libpcre_FRAMEWORKS_FOUND} ${PCRE_libpcre_SYSTEM_LIBS} ${PCRE_libpcre_DEPENDENCIES})
conan_package_library_targets("${PCRE_libpcre_LIBS}"
                              "${PCRE_libpcre_LIB_DIRS}"
                              "${PCRE_libpcre_LIBS_FRAMEWORKS_DEPS}"
                              PCRE_libpcre_NOT_USED
                              PCRE_libpcre_LIB_TARGETS
                              ""
                              "PCRE_libpcre")

set(PCRE_libpcre_LINK_LIBS ${PCRE_libpcre_LIB_TARGETS} ${PCRE_libpcre_LIBS_FRAMEWORKS_DEPS})

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

########## COMPONENT libpcreposix FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(PCRE_libpcreposix_FRAMEWORKS_FOUND "")
conan_find_apple_frameworks(PCRE_libpcreposix_FRAMEWORKS_FOUND "${PCRE_libpcreposix_FRAMEWORKS}" "${PCRE_libpcreposix_FRAMEWORK_DIRS}")

set(PCRE_libpcreposix_LIB_TARGETS "")
set(PCRE_libpcreposix_NOT_USED "")
set(PCRE_libpcreposix_LIBS_FRAMEWORKS_DEPS ${PCRE_libpcreposix_FRAMEWORKS_FOUND} ${PCRE_libpcreposix_SYSTEM_LIBS} ${PCRE_libpcreposix_DEPENDENCIES})
conan_package_library_targets("${PCRE_libpcreposix_LIBS}"
                              "${PCRE_libpcreposix_LIB_DIRS}"
                              "${PCRE_libpcreposix_LIBS_FRAMEWORKS_DEPS}"
                              PCRE_libpcreposix_NOT_USED
                              PCRE_libpcreposix_LIB_TARGETS
                              ""
                              "PCRE_libpcreposix")

set(PCRE_libpcreposix_LINK_LIBS ${PCRE_libpcreposix_LIB_TARGETS} ${PCRE_libpcreposix_LIBS_FRAMEWORKS_DEPS})

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})


########## TARGETS ##########################################################################
#############################################################################################

########## COMPONENT libpcre32 TARGET #################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    # Target approach
    if(NOT TARGET PCRE::libpcre32)
        add_library(PCRE::libpcre32 INTERFACE IMPORTED)
        set_target_properties(PCRE::libpcre32 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                              "${PCRE_libpcre32_INCLUDE_DIRS}")
        set_target_properties(PCRE::libpcre32 PROPERTIES INTERFACE_LINK_DIRECTORIES
                              "${PCRE_libpcre32_LIB_DIRS}")
        set_target_properties(PCRE::libpcre32 PROPERTIES INTERFACE_LINK_LIBRARIES
                              "${PCRE_libpcre32_LINK_LIBS};${PCRE_libpcre32_LINKER_FLAGS_LIST}")
        set_target_properties(PCRE::libpcre32 PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                              "${PCRE_libpcre32_COMPILE_DEFINITIONS}")
        set_target_properties(PCRE::libpcre32 PROPERTIES INTERFACE_COMPILE_OPTIONS
                              "${PCRE_libpcre32_COMPILE_OPTIONS_C};${PCRE_libpcre32_COMPILE_OPTIONS_CXX}")
    endif()
endif()

########## COMPONENT libpcre16 TARGET #################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    # Target approach
    if(NOT TARGET PCRE::libpcre16)
        add_library(PCRE::libpcre16 INTERFACE IMPORTED)
        set_target_properties(PCRE::libpcre16 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                              "${PCRE_libpcre16_INCLUDE_DIRS}")
        set_target_properties(PCRE::libpcre16 PROPERTIES INTERFACE_LINK_DIRECTORIES
                              "${PCRE_libpcre16_LIB_DIRS}")
        set_target_properties(PCRE::libpcre16 PROPERTIES INTERFACE_LINK_LIBRARIES
                              "${PCRE_libpcre16_LINK_LIBS};${PCRE_libpcre16_LINKER_FLAGS_LIST}")
        set_target_properties(PCRE::libpcre16 PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                              "${PCRE_libpcre16_COMPILE_DEFINITIONS}")
        set_target_properties(PCRE::libpcre16 PROPERTIES INTERFACE_COMPILE_OPTIONS
                              "${PCRE_libpcre16_COMPILE_OPTIONS_C};${PCRE_libpcre16_COMPILE_OPTIONS_CXX}")
    endif()
endif()

########## COMPONENT libpcre TARGET #################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    # Target approach
    if(NOT TARGET PCRE::libpcre)
        add_library(PCRE::libpcre INTERFACE IMPORTED)
        set_target_properties(PCRE::libpcre PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                              "${PCRE_libpcre_INCLUDE_DIRS}")
        set_target_properties(PCRE::libpcre PROPERTIES INTERFACE_LINK_DIRECTORIES
                              "${PCRE_libpcre_LIB_DIRS}")
        set_target_properties(PCRE::libpcre PROPERTIES INTERFACE_LINK_LIBRARIES
                              "${PCRE_libpcre_LINK_LIBS};${PCRE_libpcre_LINKER_FLAGS_LIST}")
        set_target_properties(PCRE::libpcre PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                              "${PCRE_libpcre_COMPILE_DEFINITIONS}")
        set_target_properties(PCRE::libpcre PROPERTIES INTERFACE_COMPILE_OPTIONS
                              "${PCRE_libpcre_COMPILE_OPTIONS_C};${PCRE_libpcre_COMPILE_OPTIONS_CXX}")
    endif()
endif()

########## COMPONENT libpcreposix TARGET #################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    # Target approach
    if(NOT TARGET PCRE::libpcreposix)
        add_library(PCRE::libpcreposix INTERFACE IMPORTED)
        set_target_properties(PCRE::libpcreposix PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                              "${PCRE_libpcreposix_INCLUDE_DIRS}")
        set_target_properties(PCRE::libpcreposix PROPERTIES INTERFACE_LINK_DIRECTORIES
                              "${PCRE_libpcreposix_LIB_DIRS}")
        set_target_properties(PCRE::libpcreposix PROPERTIES INTERFACE_LINK_LIBRARIES
                              "${PCRE_libpcreposix_LINK_LIBS};${PCRE_libpcreposix_LINKER_FLAGS_LIST}")
        set_target_properties(PCRE::libpcreposix PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                              "${PCRE_libpcreposix_COMPILE_DEFINITIONS}")
        set_target_properties(PCRE::libpcreposix PROPERTIES INTERFACE_COMPILE_OPTIONS
                              "${PCRE_libpcreposix_COMPILE_OPTIONS_C};${PCRE_libpcreposix_COMPILE_OPTIONS_CXX}")
    endif()
endif()

########## GLOBAL TARGET ####################################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    if(NOT TARGET PCRE::PCRE)
        add_library(PCRE::PCRE INTERFACE IMPORTED)
    endif()
    set_property(TARGET PCRE::PCRE APPEND PROPERTY
                 INTERFACE_LINK_LIBRARIES "${PCRE_COMPONENTS}")
endif()

########## BUILD MODULES ####################################################################
#############################################################################################
########## COMPONENT libpcre32 BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${PCRE_libpcre32_BUILD_MODULES_PATHS})
    include(${_BUILD_MODULE_PATH})
endforeach()
########## COMPONENT libpcre16 BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${PCRE_libpcre16_BUILD_MODULES_PATHS})
    include(${_BUILD_MODULE_PATH})
endforeach()
########## COMPONENT libpcre BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${PCRE_libpcre_BUILD_MODULES_PATHS})
    include(${_BUILD_MODULE_PATH})
endforeach()
########## COMPONENT libpcreposix BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${PCRE_libpcreposix_BUILD_MODULES_PATHS})
    include(${_BUILD_MODULE_PATH})
endforeach()
