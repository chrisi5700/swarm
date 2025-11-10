//
// Created by chris on 11/9/25.
//

#ifndef SWARM_COMMON_HPP
#define SWARM_COMMON_HPP
#include <SFML/Graphics.hpp>
#include <torch/torch.h>

enum class Direction
{
	Up,
	Down,
	Left,
	Right,
};

struct Accelerate
{
	Direction direction;
	float magnitude;
};

torch::Tensor to_tensor(sf::Vector2f vec);
sf::Vector2f to_vector(const torch::Tensor& tensor);
#endif // SWARM_COMMON_HPP
