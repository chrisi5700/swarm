//
// Created by chris on 11/14/25.
//

#ifndef SWARM_SIMPLEMOVINGENVIRONMENT_HPP
#define SWARM_SIMPLEMOVINGENVIRONMENT_HPP

#include <swarm/Environment.hpp>
#include <random>

struct SimpleMovingEnvironment : Environment, sf::Drawable
{
	SimpleMovingEnvironment(sf::Vector2f position, sf::Vector2f velocity, sf::Vector2f goal)
		: position(position)
		, velocity(velocity)
		, goal(goal)
	{
	}
	SimpleMovingEnvironment()
	{
		agent_shape.setFillColor(sf::Color::Green);
		agent_shape.setRadius(10);
		agent_shape.setPosition(position);

		goal_shape.setFillColor(sf::Color::Red);
		goal_shape.setSize({10, 10});
		goal_shape.setPosition(position);
	}
	sf::Vector2f position{};
	sf::Vector2f velocity{};
	sf::Vector2f goal{};
	float last_distance{};
	mutable sf::CircleShape agent_shape{};
	sf::RectangleShape goal_shape{};
	bool write_logs = false;
	torch::Tensor				 get_current_observation() const;
	StepResult					 step(const torch::Tensor& action) override;
	std::size_t					 get_observation_size() const override;
	std::size_t					 get_action_space_size() const override;
	torch::Tensor				 reset() override;
	std::unique_ptr<Environment> clone() const override;
	void set_goal(sf::Vector2f new_goal_pos);
	void toggle_log();

	 protected:
	void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
};


#endif // SWARM_SIMPLEMOVINGENVIRONMENT_HPP
