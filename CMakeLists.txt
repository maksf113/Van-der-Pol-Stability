cmake_minimum_required(VERSION 3.20)  # Minimum CMake version
project(Stability LANGUAGES CXX CUDA)  # Add CUDA language

# C++ settings
set(CMAKE_CXX_STANDARD 17)            
set(CMAKE_CXX_STANDARD_REQUIRED ON)   

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)           # Use CUDA C++17
set(CMAKE_CUDA_STANDARD_REQUIRED ON)  
set(CMAKE_CUDA_ARCHITECTURES 50)  # Target common NVIDIA GPU architectures (adjust as needed)

# Add executable from source files
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")    # C++ sources
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")    # CUDA sources
file(GLOB_RECURSE HEADER_FILES "src/*.h" "src/*.hpp")  # Header files
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${HEADER_FILES} ${CUDA_SOURCES})

# Find and link GLM, GLFW, GLEW for OpenGL
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    glfw 
    GLEW::GLEW 
    glm::glm 
    ${CUDA_LIBRARIES}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    "src"
    ${CUDA_INCLUDE_DIRS}
)

# Copy shaders to build directory
file(COPY "src/shaders/" DESTINATION "${CMAKE_BINARY_DIR}/shaders")

# Ensure CUDA files are treated as CUDA source files
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)