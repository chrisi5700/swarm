//
// Created by chris on 11/12/25.
//

#ifndef SWARM_TRAINING_HPP
#define SWARM_TRAINING_HPP
#include <swarm/Agent.hpp>
#include <swarm/Environment.hpp>
#include <memory>


struct MultiEnv :  Environment
{
	std::vector<std::unique_ptr<Environment>> envs;
	MultiEnv(std::unique_ptr<Environment> env, std::size_t num_envs)
	{
		for (std::size_t i = 0; i < num_envs-1; i++)
		{
			envs.push_back(env->clone());
		}
		envs.push_back(std::move(env));
	}
	StepResult step(const torch::Tensor& action) override
	{
		std::vector<torch::Tensor> observations;
		std::vector<torch::Tensor> rewards;
		std::vector<torch::Tensor> dones;
		observations.reserve(envs.size());
		rewards.reserve(envs.size());
		dones.reserve(envs.size());
		for (std::size_t i = 0; i < envs.size(); i++)
		{
			auto res = envs[i]->step(action[i]);

			auto done_val = res.done.item<long>();
			dones.push_back(res.done);
			rewards.push_back(res.reward);

			if (done_val != 0) {
				// episode ended — reset immediately for next timestep
				observations.push_back(envs[i]->reset());
			} else {
				observations.push_back(res.observations);
			}
		}
		return {
			torch::stack(observations),
			torch::stack(rewards),
			torch::stack(dones)
		};
	}
	std::size_t								  get_observation_size() const override
	{
		return envs[0]->get_observation_size();
	}
	std::size_t								  get_action_space_size() const override
	{
		return envs[0]->get_action_space_size();

	}
	std::unique_ptr<Environment>			  clone() const override
	{
		return nullptr; // Unused
	}
	torch::Tensor									  reset() override
	{
		std::vector<torch::Tensor> observations;
		observations.reserve(envs.size());

		for (auto& env : envs)
		{
			observations.push_back(env->reset());
		}
		return torch::stack(observations);
	}
};

struct TrainingConfig {
	long num_steps = 256;
	long num_envs = 16;
	long total_timesteps = 50000*10;
	long num_minibatches = 4;
	long update_epochs = 4;
	float learning_rate = 2.5e-4;
	float gamma = 0.99f;
	float gae_lambda = 0.95f;
	float clip_coef = 0.2f;
	float vf_coef = 0.5f;
	float ent_coef = 0.01f;
	float max_grad_norm = 0.5f;
};

void train(Agent& agent, std::unique_ptr<Environment> env, const TrainingConfig& config={});

#endif // SWARM_TRAINING_HPP