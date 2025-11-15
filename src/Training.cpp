//
// Created by chris on 11/12/25.
//
#include <swarm/Training.hpp>

void train(Agent& agent, std::unique_ptr<Environment> env, const TrainingConfig& config)
{
    MultiEnv envs(std::move(env), config.num_envs);
	auto device = torch::kCUDA;
	agent.to(device);
    torch::optim::Adam optimizer{agent.parameters(), torch::optim::AdamOptions(config.learning_rate).eps(1e-5)};

    // Storage setup
    const long obs_size = static_cast<long>(envs.get_observation_size());
    const long batch_size = config.num_steps * config.num_envs;
    const long minibatch_size = batch_size / config.num_minibatches;

    // Storage tensors
    auto obs = torch::zeros({config.num_steps, config.num_envs, obs_size}).to(device);
    auto actions = torch::zeros({config.num_steps, config.num_envs}, torch::kLong).to(device);
    auto logprobs = torch::zeros({config.num_steps, config.num_envs}).to(device);
    auto rewards = torch::zeros({config.num_steps, config.num_envs}).to(device);
    auto dones = torch::zeros({config.num_steps, config.num_envs}).to(device);
    auto values = torch::zeros({config.num_steps, config.num_envs}).to(device);

    auto next_obs = envs.reset().to(device);
    auto next_done = torch::zeros(config.num_envs).to(device);

    long num_updates = config.total_timesteps / batch_size;

    for (long update = 0; update < num_updates; update++)
    {
        // Collect rollout
        for (long step = 0; step < config.num_steps; step++)
        {
            obs[step] = next_obs;
            dones[step] = next_done;

            torch::Tensor action;
            torch::Tensor logprob;
            torch::Tensor value;
            {
                torch::NoGradGuard nograd;
                auto res = agent.get_action_and_value(next_obs);
                value = res.value.flatten();
                action = res.action;
                logprob = res.log_prob;
            }
            values[step] = value;
            actions[step] = action;
            logprobs[step] = logprob;

            auto res = envs.step(action.cpu());
            rewards[step] = res.reward.to(device).view(-1);
            next_obs = res.observations.to(device);
            next_done = res.done.to(device).view(-1);
        }
    	auto mean_reward = rewards.mean().item<float>();
    	if (update % 10 == 0) {
    		std::cout << "Update " << update
					  << " / " << num_updates
					  << "  mean reward: " << mean_reward << '\n';
    	}

        // Bootstrap value for GAE
        torch::Tensor next_value;
        {
            torch::NoGradGuard nograd;
            next_value = agent.get_value(next_obs).reshape({1, -1});
        }



        // Calculate GAE
        auto advantages = torch::zeros_like(rewards).to(device);
        torch::Tensor lastgaelam = torch::zeros({config.num_envs}).to(device); // Fix: use tensor not float

        for (long t = config.num_steps - 1; t >= 0; t--)
        {
            torch::Tensor nextnonterminal;
            torch::Tensor nextvalues;

            if (t == config.num_steps - 1)
            {
                nextnonterminal = 1.0 - next_done;
                nextvalues = next_value.squeeze(0);
            }
            else
            {
                nextnonterminal = 1.0 - dones[t + 1];
                nextvalues = values[t + 1];
            }

            auto delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t];
            advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam;
        }

        auto returns = advantages + values;

        // Flatten the batch
        auto b_obs = obs.reshape({batch_size, obs_size});
        auto b_logprobs = logprobs.reshape({batch_size});
        auto b_actions = actions.reshape({batch_size});
        auto b_advantages = advantages.reshape({batch_size});
        auto b_returns = returns.reshape({batch_size});
        auto b_values = values.reshape({batch_size});

        // PPO update
        for (long epoch = 0; epoch < config.update_epochs; epoch++)
        {
            // Generate random permutation of indices
            auto perm = torch::randperm(batch_size, torch::kLong);

            for (long start = 0; start < batch_size; start += minibatch_size)
            {
                long end = std::min(start + minibatch_size, batch_size);
                auto mb_inds = perm.slice(0, start, end).to(device);

                // Use index_select instead of index for tensor indexing
                auto res = agent.get_action_and_value(
                    b_obs.index_select(0, mb_inds),
                    b_actions.index_select(0, mb_inds)
                );
                auto newlogprob = res.log_prob;
                auto entropy = res.entropy;
                auto newvalue = res.value;

                auto logratio = newlogprob - b_logprobs.index_select(0, mb_inds);
                auto ratio = logratio.exp();

                auto mb_advantages = b_advantages.index_select(0, mb_inds);
                // Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) /
                               (mb_advantages.std() + 1e-8);

                // Policy loss
                auto pg_loss1 = -mb_advantages * ratio;
                auto pg_loss2 = -mb_advantages * torch::clamp(ratio,
                                                              1 - config.clip_coef,
                                                              1 + config.clip_coef);
                auto pg_loss = torch::max(pg_loss1, pg_loss2).mean();

                // Value loss
                newvalue = newvalue.view(-1);
                auto mb_returns = b_returns.index_select(0, mb_inds);
                auto mb_values = b_values.index_select(0, mb_inds);

                auto v_loss_unclipped = (newvalue - mb_returns).pow(2);
                auto v_clipped = mb_values +
                                torch::clamp(newvalue - mb_values,
                                           -config.clip_coef,
                                           config.clip_coef);
                auto v_loss_clipped = (v_clipped - mb_returns).pow(2);
                auto v_loss_max = torch::max(v_loss_unclipped, v_loss_clipped);
                auto v_loss = 0.5 * v_loss_max.mean();

                auto entropy_loss = entropy.mean();
                auto loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef;

                optimizer.zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(agent.parameters(), config.max_grad_norm);
                optimizer.step();
            }
        }
    }
}