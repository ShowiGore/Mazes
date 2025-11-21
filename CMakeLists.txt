cmake_minimum_required(VERSION 3.25)
project(Mazes)

set(CMAKE_CXX_STANDARD 23)

add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")


# Find the underlying C library (required by png++)
find_package(PNG REQUIRED)

# Recursively find all .cpp and .hpp files inside 'src' folder
file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp")
file(GLOB_RECURSE PROJECT_HEADERS "src/*.hpp")


# Create executable using the found files
add_executable(Mazes ${PROJECT_SOURCES} ${PROJECT_HEADERS})
#add_executable(Mazes ${SOURCE_FILES})

target_include_directories(Mazes PRIVATE src)

# Add external libraries
target_include_directories(Mazes PRIVATE
        include/png++-0.2.9  # Local header-only lib
        ${PNG_INCLUDE_DIR}   # System libpng
)

# Link dependencies
target_link_libraries(Mazes PRIVATE ${PNG_LIBRARIES})