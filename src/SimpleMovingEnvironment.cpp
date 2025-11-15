//
// Created by chris on 11/14/25.
//
#include <swarm/SimpleMovingEnvironment.hpp>

torch::Tensor SimpleMovingEnvironment::get_current_observation() const
{
	std::vector<float> observations;
	observations.push_back(velocity.x);
	observations.push_back(velocity.y);
	auto goal_dir = goal - position;
	auto dist = goal_dir.length();
	if (dist > 1e-6f) {
		goal_dir /= dist;
	} else {
		// Position basically at goal, define a safe direction
		goal_dir.x = 0.0f;
		goal_dir.y = 0.0f;
		dist = 0.0f;
	}

	float max_dist = std::sqrt(1920.0f * 1920.0f + 1080.0f * 1080.0f); // ~2200
	float norm_dist = dist / max_dist;
	observations.push_back(goal_dir.x);
	observations.push_back(goal_dir.y);
	observations.push_back(norm_dist);
	return torch::tensor(observations, torch::kFloat32);
}

Environment::StepResult SimpleMovingEnvironment::step(const torch::Tensor& action)
{
	long action_val = action.item<long>();
	velocity = {0, 0}; // Reset velocity
	switch (action_val)
	{
		case 0: velocity.x = 30.0f;
			if (write_logs)
			std::cout << "Moving Right\n";
			break;
		case 1: velocity.x = -30.0f;
			if (write_logs)
			std::cout << "Moving Left\n";
			break;
		case 2: velocity.y = 30.0f;
			if (write_logs)
			std::cout << "Moving Down\n";
			break;
		case 3: velocity.y = -30.0f;
			if (write_logs)
			std::cout << "Moving Up\n";
			break;
		default:
			throw std::invalid_argument(std::format("Unexpected action_val={}", action_val));
	}

	position += velocity;

	float new_dist = (goal - position).length();
	float reward   = last_distance - new_dist; // >0 if we moved closer
	long done      = new_dist < 10.0f;

	if (done) {
		reward += 10.0f; // terminal bonus, optional
	}

	last_distance = new_dist;

	return {
		get_current_observation(),
		torch::tensor(reward, torch::kFloat32),
		torch::tensor(done,  torch::kFloat32)
	};
}
std::size_t	  SimpleMovingEnvironment::get_observation_size() const
{
	return 5; // Vel, Direction, Dist
}
std::size_t	  SimpleMovingEnvironment::get_action_space_size() const
{
	return 4; // Right Left Up Do
}
torch::Tensor SimpleMovingEnvironment::reset()
{
	std::mt19937_64 rng{std::random_device{}()};
	std::uniform_real_distribution<float> x_dist{0, 1920};
	std::uniform_real_distribution<float> y_dist{0, 1080};
	std::uniform_real_distribution<float> vel_dis{-30, 30};

	position.x = x_dist(rng);
	position.y = y_dist(rng);
	velocity.x = vel_dis(rng);
	velocity.y = vel_dis(rng);
	goal.x = x_dist(rng);
	goal.y = y_dist(rng);

	last_distance = (goal - position).length();
	goal_shape.setPosition(goal);
	return get_current_observation();
}
std::unique_ptr<Environment> SimpleMovingEnvironment::clone() const
{
	return std::make_unique<SimpleMovingEnvironment>(*this);
}
void SimpleMovingEnvironment::set_goal(sf::Vector2f new_goal_pos)
{
	goal = new_goal_pos;
	goal_shape.setPosition(goal);
}
void SimpleMovingEnvironment::toggle_log()
{
	write_logs = !write_logs;
}
void SimpleMovingEnvironment::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
	agent_shape.setPosition(position);
	target.draw(agent_shape, states);
	target.draw(goal_shape, states);
}