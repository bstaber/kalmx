# Default target: build and run the demo
default: build run

# Configure the project (only needed once or if CMakeLists.txt changes)
configure:
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build the project
build:
    cmake --build build -j

# Run the Kalman filter demo
run:
    ./build/kf_demo

# Run tests
test:
    ctest --test-dir build

# Clean the build folder
clean:
    rm -rf build

# Rebuild from scratch
rebuild: clean configure build

# Linting and formatting
lint:
    clang-tidy src/*.cpp include/*.hpp -- -Iinclude -std=c++17
    clang-format --dry-run --Werror src/*.cpp include/*.hpp

fmt:
    clang-format -i src/*.cpp include/kalmx/*.hpp
