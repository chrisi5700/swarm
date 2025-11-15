//
// Created by chris on 11/11/25.
//

#ifndef SWARM_TENSORFACTORY_HPP
#define SWARM_TENSORFACTORY_HPP

#include <swarm/common.hpp>

class TensorFactory
{
	torch::DeviceType m_device = torch::kCPU;
	TensorFactory()
	{
		auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		if (device == torch::kCUDA)
		{
			std::cout << "Picked CUDA\n";
		} else
		{
			std::cout << "Picked CPU\n";
		}
		m_device = device;
	}
public:
	static TensorFactory& instance();
	torch::nn::Linear create_linear(std::size_t in, std::size_t out, float std=std::sqrt(2.0f)) const;
};

#endif // SWARM_TENSORFACTORY_HPP
