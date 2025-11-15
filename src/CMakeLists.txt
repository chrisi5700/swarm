# Add library definitions here
# See README.md for CMake library patterns and examples

target_add_executable(swarm)
target_sources(swarm PRIVATE common.cpp main.cpp TensorFactory.cpp Agent.cpp Training.cpp SimpleMovingEnvironment.cpp)
target_link_libraries(swarm PRIVATE
        SFML::Graphics
        SFML::Window
        ${TORCH_LIBRARIES})

target_include_directories(swarm PUBLIC ${CMAKE_SOURCE_DIR}/include)