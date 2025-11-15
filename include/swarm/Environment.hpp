//
// Created by chris on 11/12/25.
//

#ifndef SWARM_ENVIRONMENT_HPP
#define SWARM_ENVIRONMENT_HPP
#include <swarm/common.hpp>

class Environment
{
	public:
	struct StepResult
	{
		torch::Tensor observations;
		torch::Tensor reward;
		torch::Tensor done;
	};
	virtual StepResult step(const torch::Tensor& action) = 0;
	virtual std::size_t get_observation_size() const = 0;
	virtual std::size_t get_action_space_size() const = 0;
	virtual torch::Tensor reset() = 0;
	virtual std::unique_ptr<Environment> clone() const = 0;
	virtual ~Environment() = default;
};

#endif // SWARM_ENVIRONMENT_HPP
