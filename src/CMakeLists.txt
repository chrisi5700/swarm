# Add library definitions here
# See README.md for CMake library patterns and examples

target_add_executable(swarm)
target_sources(swarm PRIVATE PolicyNetwork.cpp SimulationState.cpp PPO.cpp main.cpp common.cpp)
target_link_libraries(swarm PRIVATE
        SFML::Graphics
        SFML::Window
        ${TORCH_LIBRARIES})

target_include_directories(swarm PUBLIC ${CMAKE_SOURCE_DIR}/include)