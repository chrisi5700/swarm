//
// Created by chris on 11/10/25.
//
#include <swarm/SimulationState.hpp>


int main()
{
	SimulationState simulation_state(10);
	simulation_state.train();
	simulation_state.simulate({800, 600});
}
