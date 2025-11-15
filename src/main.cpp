//
// Created by chris on 11/10/25.
//
#include <swarm/common.hpp>
#include <swarm/Agent.hpp>
#include <swarm/SimpleMovingEnvironment.hpp>
#include <swarm/Training.hpp>

template<class T>
std::ostream& operator<<(std::ostream& os, sf::Vector2<T> vec)
{
	return os << '{' << vec.x << ", " << vec.y << '}';
}

int main()
{
	auto env = std::make_unique<SimpleMovingEnvironment>();
	Agent agent{env.get()};
	train(agent, std::move(env));
	std::cout << "Done Training\n";


	agent.to(torch::kCPU);
	auto render_env = std::make_unique<SimpleMovingEnvironment>();
	render_env->reset();
	render_env->toggle_log();
	std::cout << "Reset Environment\nGoal: " << render_env->goal << "\nStarting Pos: " << render_env->position
	<< "\nStarting Vel: " << render_env->velocity << '\n';
	sf::RenderWindow window{sf::VideoMode::getDesktopMode(), "Agent", sf::Style::Default, sf::State::Fullscreen};
	while (window.isOpen())
	{
		while (auto event = window.pollEvent())
		{
			if (event->is<sf::Event::Closed>()) window.close();
			if (auto press = event->getIf<sf::Event::MouseButtonPressed>())
			{
				auto press_i = press->position;
				sf::Vector2f new_pos(press_i.x, press_i.y);
				render_env->set_goal(new_pos);
				std::cout << "Set Goal to: " << new_pos << '\n';
			}
		}
		window.clear();
		window.draw(*render_env);
		window.display();
		auto action = agent.act_greedy(render_env->get_current_observation());
		render_env->step(action);
		std::this_thread::sleep_for(std::chrono::milliseconds{500});
	}
}
