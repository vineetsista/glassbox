cmake_minimum_required(VERSION 3.22)
project(glassbox_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

if(EMSCRIPTEN)
    add_compile_options(-msimd128)
endif()

add_library(engine_core INTERFACE)
target_include_directories(engine_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(EMSCRIPTEN)
    add_executable(glassbox_wasm src/wasm_api.cpp)
    target_link_libraries(glassbox_wasm PRIVATE engine_core)
    set_target_properties(glassbox_wasm PROPERTIES
        OUTPUT_NAME "glassbox"
        SUFFIX ".js")
    target_link_options(glassbox_wasm PRIVATE
        -sMODULARIZE=1
        -sEXPORT_ES6=1
        -sEXPORT_NAME=createGlassbox
        -sALLOW_MEMORY_GROWTH=1
        -sINITIAL_MEMORY=64MB
        -sSTACK_SIZE=1MB
        -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF32,HEAP32
        -sENVIRONMENT=web,worker
        --no-entry)
else()
    add_executable(engine_cli src/main.cpp)
    target_link_libraries(engine_cli PRIVATE engine_core)

    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.4
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(engine_tests
        tests/test_ops.cpp
        tests/test_tokenizer.cpp
        tests/test_parity.cpp
    )
    target_link_libraries(engine_tests PRIVATE engine_core Catch2::Catch2WithMain)
    target_compile_definitions(engine_tests PRIVATE
        FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures")

    enable_testing()
    add_test(NAME engine_tests COMMAND engine_tests)
endif()
