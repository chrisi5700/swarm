//
// Created by chris on 11/11/25.
//
#include <swarm/TensorFactory.hpp>
TensorFactory& TensorFactory::instance()
{
	static TensorFactory instance;
	return instance;
}
torch::nn::Linear TensorFactory::create_linear(std::size_t in, std::size_t out, float std) const
{
	torch::nn::Linear linear{in, out};
	torch::nn::init::orthogonal_(linear->weight, std);
	torch::nn::init::constant_(linear->bias, 0.0);
	return linear;
}