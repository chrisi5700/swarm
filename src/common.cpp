//
// Created by chris on 11/10/25.
//
#include <swarm/common.hpp>

torch::Tensor to_tensor(sf::Vector2f vec)
{
	return torch::tensor({vec.x, vec.y}, torch::kFloat32);
}
sf::Vector2f to_vector(const torch::Tensor& tensor)
{
	return sf::Vector2f(tensor[0].item<float>(), tensor[1].item<float>());
}