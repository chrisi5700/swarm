//
// Created by chris on 11/10/25.
//
#include <swarm/PolicyNetwork.hpp>

PolicyNetwork::PolicyNetwork(std::size_t obs_size, std::size_t action_size)
	: m_obs_size(obs_size),
	  m_action_size(action_size),
	  m_shared(torch::nn::Sequential(
		  torch::nn::Linear(obs_size, 128),
		  torch::nn::ReLU(),
		  torch::nn::Linear(128, 64),
		  torch::nn::ReLU()
	  )),
	  m_mean(torch::nn::Linear(64, action_size)),
	  m_log_std(torch::nn::Linear(64, action_size)),
	  m_critic(torch::nn::Linear(64, 1))
{
	// Register all modules so they're tracked by torch
	register_module("shared", m_shared);
	register_module("mean", m_mean);
	register_module("log_std", m_log_std);
	register_module("critic", m_critic);
	for (auto& p : parameters()) {
		if (p.dim() > 1) {
			torch::nn::init::xavier_uniform_(p);
		} else {
			torch::nn::init::zeros_(p);
		}
	}
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PolicyNetwork::forward(torch::Tensor obs)
{
	auto features = m_shared->forward(obs);
	auto mean = m_mean->forward(features);
	auto log_std = m_log_std->forward(features);
	auto value = m_critic->forward(features);

	return std::make_tuple(mean, log_std, value);
}