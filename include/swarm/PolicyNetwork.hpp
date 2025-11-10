//
// Created by chris on 11/9/25.
//

#ifndef SWARM_POLICYNETWORK_HPP
#define SWARM_POLICYNETWORK_HPP

#include <torch/torch.h>

class PolicyNetwork : public torch::nn::Module
{
	std::size_t m_obs_size;
	std::size_t m_action_size;
	torch::nn::Sequential m_shared;
	torch::nn::Linear m_mean;
	torch::nn::Linear m_log_std;
	torch::nn::Linear m_critic;
public:
	PolicyNetwork(std::size_t obs_size, std::size_t action_size);
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor obs);
};

#endif // SWARM_POLICYNETWORK_HPP
