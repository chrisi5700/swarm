//
// Created by chris on 11/10/25.
//
#include <swarm/PPO.hpp>

ActionResult sample_action(torch::Tensor mean, torch::Tensor log_std) {
	// Convert log_std to std
	/*if (torch::isnan(mean).any().item<bool>() || torch::isnan(log_std).any().item<bool>()) {
		std::cout << "NaN in sample_action inputs!" << std::endl;
		std::cout << "mean: " << mean << std::endl;
		std::cout << "log_std: " << log_std << std::endl;
	}*/

	auto std = torch::exp(log_std);

	// Sample from Normal distribution - mean and std already define the shape
	auto noise = torch::randn_like(mean);  // Mean 0 Dev 1 -> Normal
	auto action = mean + std * noise;       // Reparameterization trick

	// Compute log probability
	auto log_prob = -0.5 * torch::pow((action - mean) / std, 2)
					- log_std
					- 0.5 * std::log(2 * M_PI);
	log_prob = log_prob.sum();  // Sum over action dimensions [2] -> scalar


	return {action, log_prob};
}
std::pair<torch::Tensor, torch::Tensor> compute_gae(const RolloutBuffer& buffer, std::size_t num_robots,
													std::size_t num_steps, float gamma, float lambda)
{
	std::vector<float> all_advantages;
	std::vector<float> all_returns;
	all_advantages.reserve(buffer.size());
	all_returns.reserve(buffer.size());

	// Process each robot's trajectory separately
	for (std::size_t robot = 0; robot < num_robots; ++robot)
	{
		float advantage = 0.0f;

		// Work backwards through this robot's trajectory
		for (int step = num_steps - 1; step >= 0; --step)
		{
			std::size_t idx = step * num_robots + robot;

			float next_value	= (static_cast<std::size_t>(step) == num_steps - 1) ? 0.0f : buffer.values[idx + num_robots].item<float>();
			float current_value = buffer.values[idx].item<float>();

			float delta = buffer.rewards[idx] + gamma * next_value - current_value;
			advantage	= delta + gamma * lambda * advantage;

			all_advantages.insert(all_advantages.begin(), advantage);
			all_returns.insert(all_returns.begin(), advantage + current_value);
		}
	}

	auto advantages_tensor = torch::tensor(all_advantages, torch::kFloat32);
	auto returns_tensor	   = torch::tensor(all_returns, torch::kFloat32);

	advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8f);

	return {advantages_tensor, returns_tensor};
}
void ppo_update(PolicyNetwork& network, torch::optim::Optimizer& optimizer, const RolloutBuffer& buffer,
				const torch::Tensor& advantages, const torch::Tensor& returns, int ppo_epochs, int batch_size,
				float clip_epsilon, float value_coef, float entropy_coef)
{
	std::size_t dataset_size = buffer.size();

	// Stack all tensors into batches
	auto observations  = torch::stack(buffer.observations); // [N, obs_size]
	auto actions	   = torch::stack(buffer.actions);		// [N, 2]
	auto old_log_probs = torch::stack(buffer.log_probs);	// [N]

	// Run multiple epochs over the same data
	for (int epoch = 0; epoch < ppo_epochs; ++epoch)
	{
		// Shuffle indices
		auto indices = torch::randperm(dataset_size, torch::kLong);

		// Process in mini-batches
		for (std::size_t start = 0; start < dataset_size; start += batch_size)
		{
			std::size_t end			  = std::min(start + batch_size, dataset_size);
			auto		batch_indices = indices.slice(0, start, end);

			// Extract mini-batch
			auto batch_obs			 = observations.index_select(0, batch_indices);
			auto batch_actions		 = actions.index_select(0, batch_indices);
			auto batch_old_log_probs = old_log_probs.index_select(0, batch_indices);
			auto batch_advantages	 = advantages.index_select(0, batch_indices);
			auto batch_returns		 = returns.index_select(0, batch_indices);

			// Forward pass with current network
			auto [mean, log_std, values_pred] = network.forward(batch_obs);

			// Recompute log probs with current policy
			auto std	  = torch::exp(log_std);
			auto log_prob = -0.5 * torch::pow((batch_actions - mean) / std, 2) - log_std - 0.5 * std::log(2 * M_PI);
			auto new_log_probs = log_prob.sum(1); // Sum over action dimensions

			// Compute probability ratio
			auto ratio = torch::exp(new_log_probs - batch_old_log_probs);

			// Clipped surrogate objective
			auto surr1		 = ratio * batch_advantages;
			auto surr2		 = torch::clamp(ratio, 1.0f - clip_epsilon, 1.0f + clip_epsilon) * batch_advantages;
			auto policy_loss = -torch::min(surr1, surr2).mean();

			// Value function loss
			auto value_loss = torch::mse_loss(values_pred.squeeze(), batch_returns);

			// Entropy bonus (encourages exploration)
			auto entropy = (0.5 + 0.5 * std::log(2 * M_PI) + log_std).sum(1).mean();

			// Total loss
			auto loss = policy_loss + value_coef * value_loss - entropy_coef * entropy;
			/*if (torch::isnan(loss).item<bool>() || torch::isinf(loss).item<bool>()) {
				std::cout << "NaN/Inf detected!" << std::endl;
				std::cout << "policy_loss: " << policy_loss.item<float>() << std::endl;
				std::cout << "value_loss: " << value_loss.item<float>() << std::endl;
				std::cout << "entropy: " << entropy.item<float>() << std::endl;
				std::cout << "ratio min/max: " << ratio.min().item<float>()
						  << " / " << ratio.max().item<float>() << std::endl;
				return;  // Skip this update
			}*/
			// Backpropagation
			optimizer.zero_grad();
			loss.backward();
			torch::nn::utils::clip_grad_norm_(network.parameters(), 0.5f);
			optimizer.step();
		}
	}
}
