cmake_minimum_required(VERSION 3.15)
project(DistributedTraining)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /WX-)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

add_executable(train_server server/server.cpp)

if(WIN32)
    target_link_libraries(train_server ws2_32)
else()
    target_link_libraries(train_server pthread)
endif()

# Installation rules (optional)
# install(TARGETS train_server DESTINATION bin)
# install(DIRECTORY static DESTINATION bin)
