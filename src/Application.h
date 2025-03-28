#pragma once
#include <chrono>
#include "Renderer.h"
#include "Window.h"
#include "CudaProcessor.h"
#include "Keyboard.h"
#include "Mouse.h"

class Application
{
private:
	Window m_window;
	Renderer m_renderer;
	CudaProcessor m_cudaProcessor;
	double m_deltaParam = 0.5;
	float m_param = 0.1f;
public:
	Application(int width, int height);
	~Application() = default;
	void run();
private:
	void handleInput(double deltaTime);
	void setWindowTitle();
	void printInstructions();
};

Application::Application(int width, int height) : m_window(width, height), m_renderer(width, height), m_cudaProcessor(width, height) 
{
	printInstructions();
}

inline void Application::run()
{
	auto lastTime = std::chrono::high_resolution_clock::now();

	while (!m_window.shouldClose())
	{
		auto crntTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> deltaTime = crntTime - lastTime; // in seconds
		lastTime = crntTime;
		m_window.pollEvents();
		handleInput(deltaTime.count());
		setWindowTitle();
		m_cudaProcessor.process(m_param);
		m_cudaProcessor.bindPBO();
		m_renderer.render();
		m_window.swapBuffers();

	}
}

inline void Application::handleInput(double deltaTime)
{
	if (Keyboard::key(GLFW_KEY_UP))
		m_param += m_deltaParam * deltaTime;
	if (Keyboard::key(GLFW_KEY_DOWN))
		m_param -= m_deltaParam * deltaTime;
}

inline void Application::setWindowTitle()
{
	std::string title = "Van der Pol Oscilator, param = " + std::to_string(m_param);
	m_window.setTitle(title.c_str());
}

void Application::printInstructions()
{
	printf("Stability visualizer\n");
	printf("Use up/down arrow keys adjust parameter value\n\n");
}
