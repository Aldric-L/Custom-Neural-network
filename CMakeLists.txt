
cmake_minimum_required(VERSION 3.27)
project(AKML_Sandbox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

set(SOURCES main.cpp)
add_executable(AKML_Sandbox ${SOURCES})

add_subdirectory(v0.2)
target_link_libraries(AKML_Sandbox PRIVATE AKML)
target_include_directories(AKML_Sandbox PRIVATE v0.2)
set_property(TARGET AKML PROPERTY CXX_STANDARD 20)

