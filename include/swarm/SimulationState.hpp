//
// Created by chris on 11/9/25.
//

#ifndef SWARM_SIMULATIONSTATE_HPP
#define SWARM_SIMULATIONSTATE_HPP
#include <memory>
#include <SFML/System/Vector2.hpp>
#include <vector>

#include "PolicyNetwork.hpp"
#include "PPO.hpp"

class SimulationState
{
	std::size_t					  m_size;
	std::unique_ptr<sf::Vector2f> m_pos;
	std::unique_ptr<sf::Vector2f> m_vel;
	float						  m_dt;
	static constexpr std::size_t  NEAREST_NEIGHBOUR_COUNT = 5;
	static constexpr std::size_t  INPUT_VECTOR_COUNT	  = NEAREST_NEIGHBOUR_COUNT * 2 + 1 + 1;
	// Nearest neighbour speed/vel, own speed, goal
	PolicyNetwork m_policy_network{INPUT_VECTOR_COUNT * 2, 2}; // 2 floats per vector

	 public:
	explicit SimulationState(std::size_t size);
	void		  step_simulation(const std::vector<sf::Vector2f>&);
	torch::Tensor get_observation(std::size_t idx, const sf::Vector2f& goal);
	float		  compute_reward(std::size_t idx, const sf::Vector2f& goal);
	void		  collect_rollout(int num_steps, const sf::Vector2f& goal, RolloutBuffer& buffer);
	void		  reset();
	void		  train(int num_episodes = 1000, int steps_per_episode = 500, float learning_rate = 3e-4f);
	void		  simulate(sf::Vector2f goal);
	void		  draw_to(sf::RenderWindow&);
};

#endif // SWARM_SIMULATIONSTATE_HPP
