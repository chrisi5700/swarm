//
// Created by chris on 11/12/25.
//

#ifndef SWARM_AGENT_HPP
#define SWARM_AGENT_HPP
#include <swarm/TensorFactory.hpp>

#include "Environment.hpp"

class Agent : public torch::nn::Module
{
private:
	torch::nn::Sequential m_critic;
	torch::nn::Sequential m_actor;

public:
	struct ActionDetails
	{
		torch::Tensor action;
		torch::Tensor log_prob;
		torch::Tensor entropy;
		torch::Tensor value;
	};
	Agent(const Environment* env)
	{
		m_critic = register_module("critic", torch::nn::Sequential(
			TensorFactory::instance().create_linear(env->get_observation_size(), 64),
			torch::nn::Tanh(),
			TensorFactory::instance().create_linear(64, 64),
			torch::nn::Tanh(),
			TensorFactory::instance().create_linear(64, 1, 1.0)
		));

		m_actor = register_module("actor", torch::nn::Sequential(
			TensorFactory::instance().create_linear(env->get_observation_size(), 64),
			torch::nn::Tanh(),
			TensorFactory::instance().create_linear(64, 64),
			torch::nn::Tanh(),
			TensorFactory::instance().create_linear(64, env->get_action_space_size(), 0.01)
		));
	}

	torch::Tensor get_value(const torch::Tensor& observation)
	{
		return m_critic->forward(observation);
	}

	ActionDetails get_action_and_value(const torch::Tensor& observation,
								  std::optional<torch::Tensor> action = std::nullopt)
	{
		auto logits = m_actor->forward(observation);

		// Use Categorical distribution approach (more similar to Python)
		auto probs = torch::softmax(logits, -1);

		torch::Tensor sampled_action;
		if (!action.has_value()) {
			// Sample from categorical distribution
			sampled_action = torch::multinomial(probs, 1, /*replacement=*/true).squeeze(-1);
		} else {
			sampled_action = action.value();
		}

		// Get log probability using log_softmax directly on logits (more numerically stable)
		auto log_probs = torch::log_softmax(logits, -1);
		auto action_log_prob = log_probs.gather(-1, sampled_action.unsqueeze(-1)).squeeze(-1);

		// Calculate entropy
		auto entropy = -(probs * log_probs).sum(-1);

		// Get value - note: don't squeeze here yet, let the caller decide
		auto value = m_critic->forward(observation);

		return {sampled_action, action_log_prob, entropy, value};
	}
	torch::Tensor act_greedy(const torch::Tensor& observation)
	{
		auto logits = m_actor->forward(observation);
		auto probs  = torch::softmax(logits, -1);
		auto result = probs.argmax(-1);
		return result;
	}

};

#endif // SWARM_AGENT_HPP
