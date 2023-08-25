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

conan_message(STATUS "Conan: Using autogenerated FindPostgreSQL.cmake")
set(PostgreSQL_FOUND 1)
set(PostgreSQL_VERSION "14.7")

find_package_handle_standard_args(PostgreSQL REQUIRED_VARS
                                  PostgreSQL_VERSION VERSION_VAR PostgreSQL_VERSION)
mark_as_advanced(PostgreSQL_FOUND PostgreSQL_VERSION)

set(PostgreSQL_COMPONENTS PostgreSQL::pq PostgreSQL::pgcommon PostgreSQL::pgport)

if(PostgreSQL_FIND_COMPONENTS)
    foreach(_FIND_COMPONENT ${PostgreSQL_FIND_COMPONENTS})
        list(FIND PostgreSQL_COMPONENTS "PostgreSQL::${_FIND_COMPONENT}" _index)
        if(${_index} EQUAL -1)
            conan_message(FATAL_ERROR "Conan: Component '${_FIND_COMPONENT}' NOT found in package 'PostgreSQL'")
        else()
            conan_message(STATUS "Conan: Component '${_FIND_COMPONENT}' found in package 'PostgreSQL'")
        endif()
    endforeach()
endif()

########### VARIABLES #######################################################################
#############################################################################################


set(PostgreSQL_INCLUDE_DIRS "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_INCLUDE_DIR "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_INCLUDES "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_RES_DIRS )
set(PostgreSQL_DEFINITIONS )
set(PostgreSQL_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)
set(PostgreSQL_COMPILE_DEFINITIONS )
set(PostgreSQL_COMPILE_OPTIONS_LIST "" "")
set(PostgreSQL_COMPILE_OPTIONS_C "")
set(PostgreSQL_COMPILE_OPTIONS_CXX "")
set(PostgreSQL_LIBRARIES_TARGETS "") # Will be filled later, if CMake 3
set(PostgreSQL_LIBRARIES "") # Will be filled later
set(PostgreSQL_LIBS "") # Same as PostgreSQL_LIBRARIES
set(PostgreSQL_SYSTEM_LIBS pthread)
set(PostgreSQL_FRAMEWORK_DIRS )
set(PostgreSQL_FRAMEWORKS )
set(PostgreSQL_FRAMEWORKS_FOUND "") # Will be filled later
set(PostgreSQL_BUILD_MODULES_PATHS )

conan_find_apple_frameworks(PostgreSQL_FRAMEWORKS_FOUND "${PostgreSQL_FRAMEWORKS}" "${PostgreSQL_FRAMEWORK_DIRS}")

mark_as_advanced(PostgreSQL_INCLUDE_DIRS
                 PostgreSQL_INCLUDE_DIR
                 PostgreSQL_INCLUDES
                 PostgreSQL_DEFINITIONS
                 PostgreSQL_LINKER_FLAGS_LIST
                 PostgreSQL_COMPILE_DEFINITIONS
                 PostgreSQL_COMPILE_OPTIONS_LIST
                 PostgreSQL_LIBRARIES
                 PostgreSQL_LIBS
                 PostgreSQL_LIBRARIES_TARGETS)

# Find the real .lib/.a and add them to PostgreSQL_LIBS and PostgreSQL_LIBRARY_LIST
set(PostgreSQL_LIBRARY_LIST pq pgcommon pgcommon_shlib pgport pgport_shlib)
set(PostgreSQL_LIB_DIRS "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")

# Gather all the libraries that should be linked to the targets (do not touch existing variables):
set(_PostgreSQL_DEPENDENCIES "${PostgreSQL_FRAMEWORKS_FOUND} ${PostgreSQL_SYSTEM_LIBS} ")

conan_package_library_targets("${PostgreSQL_LIBRARY_LIST}"  # libraries
                              "${PostgreSQL_LIB_DIRS}"      # package_libdir
                              "${_PostgreSQL_DEPENDENCIES}"  # deps
                              PostgreSQL_LIBRARIES            # out_libraries
                              PostgreSQL_LIBRARIES_TARGETS    # out_libraries_targets
                              ""                          # build_type
                              "PostgreSQL")                                      # package_name

set(PostgreSQL_LIBS ${PostgreSQL_LIBRARIES})

foreach(_FRAMEWORK ${PostgreSQL_FRAMEWORKS_FOUND})
    list(APPEND PostgreSQL_LIBRARIES_TARGETS ${_FRAMEWORK})
    list(APPEND PostgreSQL_LIBRARIES ${_FRAMEWORK})
endforeach()

foreach(_SYSTEM_LIB ${PostgreSQL_SYSTEM_LIBS})
    list(APPEND PostgreSQL_LIBRARIES_TARGETS ${_SYSTEM_LIB})
    list(APPEND PostgreSQL_LIBRARIES ${_SYSTEM_LIB})
endforeach()

# We need to add our requirements too
set(PostgreSQL_LIBRARIES_TARGETS "${PostgreSQL_LIBRARIES_TARGETS};")
set(PostgreSQL_LIBRARIES "${PostgreSQL_LIBRARIES};")

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})


########### COMPONENT pgport VARIABLES #############################################

set(PostgreSQL_pgport_INCLUDE_DIRS "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pgport_INCLUDE_DIR "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pgport_INCLUDES "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pgport_LIB_DIRS "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(PostgreSQL_pgport_RES_DIRS )
set(PostgreSQL_pgport_DEFINITIONS )
set(PostgreSQL_pgport_COMPILE_DEFINITIONS )
set(PostgreSQL_pgport_COMPILE_OPTIONS_C "")
set(PostgreSQL_pgport_COMPILE_OPTIONS_CXX "")
set(PostgreSQL_pgport_LIBS pgport pgport_shlib)
set(PostgreSQL_pgport_SYSTEM_LIBS )
set(PostgreSQL_pgport_FRAMEWORK_DIRS )
set(PostgreSQL_pgport_FRAMEWORKS )
set(PostgreSQL_pgport_BUILD_MODULES_PATHS )
set(PostgreSQL_pgport_DEPENDENCIES )
set(PostgreSQL_pgport_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)

########### COMPONENT pgcommon VARIABLES #############################################

set(PostgreSQL_pgcommon_INCLUDE_DIRS "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pgcommon_INCLUDE_DIR "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pgcommon_INCLUDES "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pgcommon_LIB_DIRS "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(PostgreSQL_pgcommon_RES_DIRS )
set(PostgreSQL_pgcommon_DEFINITIONS )
set(PostgreSQL_pgcommon_COMPILE_DEFINITIONS )
set(PostgreSQL_pgcommon_COMPILE_OPTIONS_C "")
set(PostgreSQL_pgcommon_COMPILE_OPTIONS_CXX "")
set(PostgreSQL_pgcommon_LIBS pgcommon pgcommon_shlib)
set(PostgreSQL_pgcommon_SYSTEM_LIBS )
set(PostgreSQL_pgcommon_FRAMEWORK_DIRS )
set(PostgreSQL_pgcommon_FRAMEWORKS )
set(PostgreSQL_pgcommon_BUILD_MODULES_PATHS )
set(PostgreSQL_pgcommon_DEPENDENCIES PostgreSQL::pgport)
set(PostgreSQL_pgcommon_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)

########### COMPONENT pq VARIABLES #############################################

set(PostgreSQL_pq_INCLUDE_DIRS "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pq_INCLUDE_DIR "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pq_INCLUDES "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/include")
set(PostgreSQL_pq_LIB_DIRS "/home/blackbird/.conan/data/libpq/14.7/_/_/package/6af9cc7cb931c5ad942174fd7838eb655717c709/lib")
set(PostgreSQL_pq_RES_DIRS )
set(PostgreSQL_pq_DEFINITIONS )
set(PostgreSQL_pq_COMPILE_DEFINITIONS )
set(PostgreSQL_pq_COMPILE_OPTIONS_C "")
set(PostgreSQL_pq_COMPILE_OPTIONS_CXX "")
set(PostgreSQL_pq_LIBS pq)
set(PostgreSQL_pq_SYSTEM_LIBS pthread)
set(PostgreSQL_pq_FRAMEWORK_DIRS )
set(PostgreSQL_pq_FRAMEWORKS )
set(PostgreSQL_pq_BUILD_MODULES_PATHS )
set(PostgreSQL_pq_DEPENDENCIES PostgreSQL::pgcommon)
set(PostgreSQL_pq_LINKER_FLAGS_LIST
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,SHARED_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,MODULE_LIBRARY>:>"
        "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:>"
)


########## FIND PACKAGE DEPENDENCY ##########################################################
#############################################################################################

include(CMakeFindDependencyMacro)


########## FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #######################################
#############################################################################################

########## COMPONENT pgport FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(PostgreSQL_pgport_FRAMEWORKS_FOUND "")
conan_find_apple_frameworks(PostgreSQL_pgport_FRAMEWORKS_FOUND "${PostgreSQL_pgport_FRAMEWORKS}" "${PostgreSQL_pgport_FRAMEWORK_DIRS}")

set(PostgreSQL_pgport_LIB_TARGETS "")
set(PostgreSQL_pgport_NOT_USED "")
set(PostgreSQL_pgport_LIBS_FRAMEWORKS_DEPS ${PostgreSQL_pgport_FRAMEWORKS_FOUND} ${PostgreSQL_pgport_SYSTEM_LIBS} ${PostgreSQL_pgport_DEPENDENCIES})
conan_package_library_targets("${PostgreSQL_pgport_LIBS}"
                              "${PostgreSQL_pgport_LIB_DIRS}"
                              "${PostgreSQL_pgport_LIBS_FRAMEWORKS_DEPS}"
                              PostgreSQL_pgport_NOT_USED
                              PostgreSQL_pgport_LIB_TARGETS
                              ""
                              "PostgreSQL_pgport")

set(PostgreSQL_pgport_LINK_LIBS ${PostgreSQL_pgport_LIB_TARGETS} ${PostgreSQL_pgport_LIBS_FRAMEWORKS_DEPS})

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

########## COMPONENT pgcommon FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(PostgreSQL_pgcommon_FRAMEWORKS_FOUND "")
conan_find_apple_frameworks(PostgreSQL_pgcommon_FRAMEWORKS_FOUND "${PostgreSQL_pgcommon_FRAMEWORKS}" "${PostgreSQL_pgcommon_FRAMEWORK_DIRS}")

set(PostgreSQL_pgcommon_LIB_TARGETS "")
set(PostgreSQL_pgcommon_NOT_USED "")
set(PostgreSQL_pgcommon_LIBS_FRAMEWORKS_DEPS ${PostgreSQL_pgcommon_FRAMEWORKS_FOUND} ${PostgreSQL_pgcommon_SYSTEM_LIBS} ${PostgreSQL_pgcommon_DEPENDENCIES})
conan_package_library_targets("${PostgreSQL_pgcommon_LIBS}"
                              "${PostgreSQL_pgcommon_LIB_DIRS}"
                              "${PostgreSQL_pgcommon_LIBS_FRAMEWORKS_DEPS}"
                              PostgreSQL_pgcommon_NOT_USED
                              PostgreSQL_pgcommon_LIB_TARGETS
                              ""
                              "PostgreSQL_pgcommon")

set(PostgreSQL_pgcommon_LINK_LIBS ${PostgreSQL_pgcommon_LIB_TARGETS} ${PostgreSQL_pgcommon_LIBS_FRAMEWORKS_DEPS})

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})

########## COMPONENT pq FIND LIBRARIES & FRAMEWORKS / DYNAMIC VARS #############

set(PostgreSQL_pq_FRAMEWORKS_FOUND "")
conan_find_apple_frameworks(PostgreSQL_pq_FRAMEWORKS_FOUND "${PostgreSQL_pq_FRAMEWORKS}" "${PostgreSQL_pq_FRAMEWORK_DIRS}")

set(PostgreSQL_pq_LIB_TARGETS "")
set(PostgreSQL_pq_NOT_USED "")
set(PostgreSQL_pq_LIBS_FRAMEWORKS_DEPS ${PostgreSQL_pq_FRAMEWORKS_FOUND} ${PostgreSQL_pq_SYSTEM_LIBS} ${PostgreSQL_pq_DEPENDENCIES})
conan_package_library_targets("${PostgreSQL_pq_LIBS}"
                              "${PostgreSQL_pq_LIB_DIRS}"
                              "${PostgreSQL_pq_LIBS_FRAMEWORKS_DEPS}"
                              PostgreSQL_pq_NOT_USED
                              PostgreSQL_pq_LIB_TARGETS
                              ""
                              "PostgreSQL_pq")

set(PostgreSQL_pq_LINK_LIBS ${PostgreSQL_pq_LIB_TARGETS} ${PostgreSQL_pq_LIBS_FRAMEWORKS_DEPS})

set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH  ${CMAKE_PREFIX_PATH})


########## TARGETS ##########################################################################
#############################################################################################

########## COMPONENT pgport TARGET #################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    # Target approach
    if(NOT TARGET PostgreSQL::pgport)
        add_library(PostgreSQL::pgport INTERFACE IMPORTED)
        set_target_properties(PostgreSQL::pgport PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                              "${PostgreSQL_pgport_INCLUDE_DIRS}")
        set_target_properties(PostgreSQL::pgport PROPERTIES INTERFACE_LINK_DIRECTORIES
                              "${PostgreSQL_pgport_LIB_DIRS}")
        set_target_properties(PostgreSQL::pgport PROPERTIES INTERFACE_LINK_LIBRARIES
                              "${PostgreSQL_pgport_LINK_LIBS};${PostgreSQL_pgport_LINKER_FLAGS_LIST}")
        set_target_properties(PostgreSQL::pgport PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                              "${PostgreSQL_pgport_COMPILE_DEFINITIONS}")
        set_target_properties(PostgreSQL::pgport PROPERTIES INTERFACE_COMPILE_OPTIONS
                              "${PostgreSQL_pgport_COMPILE_OPTIONS_C};${PostgreSQL_pgport_COMPILE_OPTIONS_CXX}")
    endif()
endif()

########## COMPONENT pgcommon TARGET #################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    # Target approach
    if(NOT TARGET PostgreSQL::pgcommon)
        add_library(PostgreSQL::pgcommon INTERFACE IMPORTED)
        set_target_properties(PostgreSQL::pgcommon PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                              "${PostgreSQL_pgcommon_INCLUDE_DIRS}")
        set_target_properties(PostgreSQL::pgcommon PROPERTIES INTERFACE_LINK_DIRECTORIES
                              "${PostgreSQL_pgcommon_LIB_DIRS}")
        set_target_properties(PostgreSQL::pgcommon PROPERTIES INTERFACE_LINK_LIBRARIES
                              "${PostgreSQL_pgcommon_LINK_LIBS};${PostgreSQL_pgcommon_LINKER_FLAGS_LIST}")
        set_target_properties(PostgreSQL::pgcommon PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                              "${PostgreSQL_pgcommon_COMPILE_DEFINITIONS}")
        set_target_properties(PostgreSQL::pgcommon PROPERTIES INTERFACE_COMPILE_OPTIONS
                              "${PostgreSQL_pgcommon_COMPILE_OPTIONS_C};${PostgreSQL_pgcommon_COMPILE_OPTIONS_CXX}")
    endif()
endif()

########## COMPONENT pq TARGET #################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    # Target approach
    if(NOT TARGET PostgreSQL::pq)
        add_library(PostgreSQL::pq INTERFACE IMPORTED)
        set_target_properties(PostgreSQL::pq PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                              "${PostgreSQL_pq_INCLUDE_DIRS}")
        set_target_properties(PostgreSQL::pq PROPERTIES INTERFACE_LINK_DIRECTORIES
                              "${PostgreSQL_pq_LIB_DIRS}")
        set_target_properties(PostgreSQL::pq PROPERTIES INTERFACE_LINK_LIBRARIES
                              "${PostgreSQL_pq_LINK_LIBS};${PostgreSQL_pq_LINKER_FLAGS_LIST}")
        set_target_properties(PostgreSQL::pq PROPERTIES INTERFACE_COMPILE_DEFINITIONS
                              "${PostgreSQL_pq_COMPILE_DEFINITIONS}")
        set_target_properties(PostgreSQL::pq PROPERTIES INTERFACE_COMPILE_OPTIONS
                              "${PostgreSQL_pq_COMPILE_OPTIONS_C};${PostgreSQL_pq_COMPILE_OPTIONS_CXX}")
    endif()
endif()

########## GLOBAL TARGET ####################################################################

if(NOT ${CMAKE_VERSION} VERSION_LESS "3.0")
    if(NOT TARGET PostgreSQL::PostgreSQL)
        add_library(PostgreSQL::PostgreSQL INTERFACE IMPORTED)
    endif()
    set_property(TARGET PostgreSQL::PostgreSQL APPEND PROPERTY
                 INTERFACE_LINK_LIBRARIES "${PostgreSQL_COMPONENTS}")
endif()

########## BUILD MODULES ####################################################################
#############################################################################################
########## COMPONENT pgport BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${PostgreSQL_pgport_BUILD_MODULES_PATHS})
    include(${_BUILD_MODULE_PATH})
endforeach()
########## COMPONENT pgcommon BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${PostgreSQL_pgcommon_BUILD_MODULES_PATHS})
    include(${_BUILD_MODULE_PATH})
endforeach()
########## COMPONENT pq BUILD MODULES ##########################################

foreach(_BUILD_MODULE_PATH ${PostgreSQL_pq_BUILD_MODULES_PATHS})
    include(${_BUILD_MODULE_PATH})
endforeach()
