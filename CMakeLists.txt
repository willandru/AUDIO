cmake_minimum_required(VERSION 3.20)

project(AUDIO LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

add_subdirectory(
    externals/portaudio
    ${CMAKE_BINARY_DIR}/portaudio
    EXCLUDE_FROM_ALL
)

add_executable(AUDIO
    src/main.cpp
)

target_link_libraries(AUDIO PRIVATE portaudio)

add_custom_command(TARGET AUDIO POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:portaudio>
        $<TARGET_FILE_DIR:AUDIO>
)