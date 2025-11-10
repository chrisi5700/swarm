//
// Created by chris on 11/9/25.
//
#include <random>
#include <swarm/SimulationState.hpp>

#include "swarm/PPO.hpp"

SimulationState::SimulationState(std::size_t size)
	: m_size(size)
	, m_pos(new sf::Vector2f[m_size])
	, m_vel(new sf::Vector2f[m_size])
	, m_dt(0.1f)
{
	reset();
}
void SimulationState::step_simulation(const std::vector<sf::Vector2f>& accel)
{
	for (std::size_t i = 0; i < m_size; ++i)
	{
		m_vel.get()[i] += accel[i] * m_dt;
		m_pos.get()[i] += m_vel.get()[i] * m_dt;
		m_vel.get()[i] *= 0.95f; // dampening
	}
}
torch::Tensor SimulationState::get_observation(std::size_t idx, const sf::Vector2f& goal)
{
	assert(m_size > NEAREST_NEIGHBOUR_COUNT && "Not enough robots for K neighbors");
	/*if (std::isnan(m_pos.get()[idx].x) || std::isnan(m_pos.get()[idx].y)) {
		std::cout << "NaN position for robot " << idx << std::endl;
	}
	if (std::isnan(goal.x) || std::isnan(goal.y)) {
		std::cout << "NaN goal position" << std::endl;
	}*/
	std::array<sf::Vector2f, INPUT_VECTOR_COUNT> data;
	data[0] = m_vel.get()[idx];
	data[1] = goal - m_pos.get()[idx]; // vector TO goal

	// Find K nearest neighbors
	std::vector<std::pair<float, std::size_t>> distances; // (distance_sq, index)
	distances.reserve(m_size - 1);

	for (std::size_t i = 0; i < m_size; ++i)
	{
		if (i == idx)
			continue; // Skip self

		sf::Vector2f diff	 = m_pos.get()[i] - m_pos.get()[idx];
		float		 dist_sq = diff.x * diff.x + diff.y * diff.y;
		distances.emplace_back(dist_sq, i);
	}

	// Partial sort to get K smallest
	std::partial_sort(distances.begin(), distances.begin() + NEAREST_NEIGHBOUR_COUNT, distances.end(),
					  [](const auto& a, const auto& b) { return a.first < b.first; });

	// Fill in neighbor data
	for (std::size_t k = 0; k < NEAREST_NEIGHBOUR_COUNT; ++k)
	{
		std::size_t neighbor_idx = distances[k].second;
		// Relative position
		data[2 + k * 2] = m_pos.get()[neighbor_idx] - m_pos.get()[idx];
		// Relative velocity
		data[2 + k * 2 + 1] = m_vel.get()[neighbor_idx] - m_vel.get()[idx];
	}

	// Convert to flat tensor
	std::array<float, INPUT_VECTOR_COUNT * 2> flat_data{};
	std::size_t								  i = 0;
	for (const auto& vec : data)
	{
		flat_data[i++] = vec.x;
		flat_data[i++] = vec.y;
	}

	return torch::from_blob(flat_data.data(), {INPUT_VECTOR_COUNT * 2}, torch::kFloat32).clone();
}

float SimulationState::compute_reward(std::size_t idx, const sf::Vector2f& goal)
{
	// Distance to goal (negative reward)
	sf::Vector2f diff		  = m_pos.get()[idx] - goal;
	float		 dist_to_goal = diff.length();
	float		 reward		  = -dist_to_goal + 50;

	// Check for collisions with other robots
	constexpr float collision_threshold = 5.0f; // Adjust based on your robot size
	constexpr float collision_penalty	= -10.0f;

	for (std::size_t i = 0; i < m_size; ++i)
	{
		if (i == idx)
			continue;

		sf::Vector2f robot_diff = m_pos.get()[i] - m_pos.get()[idx];
		float		 dist_sq	= robot_diff.lengthSquared();

		if (dist_sq < collision_threshold * collision_threshold)
		{
			reward += collision_penalty;
			break; // Only penalize once per timestep
		}
	}
	reward = std::clamp(reward, -100.0f, 100.0f);
	return reward;
}

void SimulationState::collect_rollout(int num_steps, const sf::Vector2f& goal, RolloutBuffer& buffer)
{
	buffer.clear();

	// Reset simulation - spawn robots at random positions
	reset();

	// Collect experience for num_steps timesteps
	for (int step = 0; step < num_steps; ++step)
	{
		std::vector<sf::Vector2f> accelerations;
		accelerations.reserve(m_size);

		// For each robot, get observation and sample action
		for (std::size_t i = 0; i < m_size; ++i)
		{
			// Get observation
			auto obs = get_observation(i, goal);

			// Forward pass through policy network
			auto [mean, log_std, value] = m_policy_network.forward(obs);

			// Sample action
			auto [action, log_prob] = sample_action(mean, log_std);

			// Compute reward
			float reward = compute_reward(i, goal);

			// Store in buffer
			buffer.observations.push_back(obs.detach());
			buffer.actions.push_back(action.detach());
			buffer.log_probs.push_back(log_prob.detach());
			buffer.rewards.push_back(reward);
			buffer.values.push_back(value.detach());

			// Convert action tensor to sf::Vector2f for simulation
			auto action_vec = to_vector(action);
			constexpr float max_accel = 10.0f;
			action_vec.x = std::clamp(action_vec.x, -max_accel, max_accel);
			action_vec.y = std::clamp(action_vec.y, -max_accel, max_accel);
			accelerations.push_back(action_vec);
		}

		// Step the simulation with all actions
		step_simulation(accelerations);
	}
}
void SimulationState::reset()
{
	// Random number generator
	static std::mt19937					  gen(std::random_device{}());
	std::uniform_real_distribution<float> dist_x(0.0f, 1920.0f);
	std::uniform_real_distribution<float> dist_y(0.0f, 1080.0f);

	// Spawn robots at random positions with zero velocity
	for (std::size_t i = 0; i < m_size; ++i)
	{
		m_pos.get()[i] = sf::Vector2f(dist_x(gen), dist_y(gen));
		m_vel.get()[i] = sf::Vector2f(0.0f, 0.0f);
	}
}
void SimulationState::train(int num_episodes, int steps_per_episode, float learning_rate)
{
	// Create optimizer
	torch::optim::Adam optimizer(m_policy_network.parameters(), torch::optim::AdamOptions(learning_rate));

	// Create rollout buffer
	RolloutBuffer buffer;

	// Training loop
	for (int episode = 0; episode < num_episodes; ++episode)
	{
		// Random goal for this episode
		std::random_device					  rd;
		std::mt19937						  gen(rd());
		std::uniform_real_distribution<float> dist_x(0.0f, 800.0f);
		std::uniform_real_distribution<float> dist_y(0.0f, 600.0f);
		sf::Vector2f						  goal(dist_x(gen), dist_y(gen));

		// Collect rollout
		collect_rollout(steps_per_episode, goal, buffer);

		// Compute advantages and returns
		auto [advantages, returns] = compute_gae(buffer,
												 m_size, // You'll need to add this getter
												 steps_per_episode);

		// PPO update
		ppo_update(m_policy_network, optimizer, buffer, advantages, returns);

		// Logging (optional but useful)
		if (episode % 10 == 0)
		{
			float avg_reward = 0.0f;
			for (float r : buffer.rewards)
			{
				avg_reward += r;
			}
			avg_reward /= buffer.rewards.size();

			std::cout << "Episode " << episode << " | Avg Reward: " << avg_reward << std::endl;
		}
	}
}
void SimulationState::simulate(sf::Vector2f goal)
{
	std::vector<sf::Vector2f> accelerations;
	accelerations.reserve(m_size);
	sf::RenderWindow window{sf::VideoMode{{1920, 1080}}, "Swarm"};
	auto last_frame = std::chrono::steady_clock::now();
	while (window.isOpen())
	{
		// For each robot, get observation and sample action
		for (std::size_t i = 0; i < m_size; ++i)
		{
			// Get observation
			auto obs = get_observation(i, goal);

			// Forward pass through policy network
			auto [mean, log_std, value] = m_policy_network.forward(obs);


			accelerations.push_back(to_vector(mean));
		}

		while (auto event = window.pollEvent())
		{
			if (event->is<sf::Event::Closed>())
				window.close();
		}

		// Step the simulation with all actions
		step_simulation(accelerations);
		accelerations.clear();
		draw_to(window);
		window.display();
		std::this_thread::sleep_until(last_frame + std::chrono::duration<float>(m_dt));
		last_frame = std::chrono::steady_clock::now(); // Tied to FPS but oh well
	}
}
void SimulationState::draw_to(sf::RenderWindow& window)
{
	static sf::CircleShape shape = []()
	{
		sf::CircleShape out;
		out.setFillColor(sf::Color::Green);
		out.setRadius(5.0f);
		return out;
	}();
	for (std::size_t i = 0; i < m_size; ++i)
	{
		shape.setPosition(m_pos.get()[i]);
		window.draw(shape);
	}
}