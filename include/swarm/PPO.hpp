//
// Created by chris on 11/10/25.
//

#ifndef SWARM_PPO_HPP
#define SWARM_PPO_HPP
#include <swarm/common.hpp>

#include "PolicyNetwork.hpp"

struct ActionResult {
	torch::Tensor action{};      // The sampled action [ax, ay]
	torch::Tensor log_prob{};    // Log probability of this action
};

ActionResult sample_action(torch::Tensor mean, torch::Tensor log_std);


struct RolloutBuffer {
	std::vector<torch::Tensor> observations;
	std::vector<torch::Tensor> actions;
	std::vector<torch::Tensor> log_probs;
	std::vector<float> rewards;
	std::vector<torch::Tensor> values;

	void clear() {
		observations.clear();
		actions.clear();
		log_probs.clear();
		rewards.clear();
		values.clear();
	}

	std::size_t size() const {
		return observations.size();
	}
};

std::pair<torch::Tensor, torch::Tensor> compute_gae(
	const RolloutBuffer& buffer,
	std::size_t num_robots,
	std::size_t num_steps,
	float gamma = 0.99f,
	float lambda = 0.95f
);

void ppo_update(
    PolicyNetwork& network,
    torch::optim::Optimizer& optimizer,
    const RolloutBuffer& buffer,
    const torch::Tensor& advantages,
    const torch::Tensor& returns,
    int ppo_epochs = 10,
    int batch_size = 64,
    float clip_epsilon = 0.2f,
    float value_coef = 0.5f,
    float entropy_coef = 0.01f
);



#endif // SWARM_PPO_HPP
