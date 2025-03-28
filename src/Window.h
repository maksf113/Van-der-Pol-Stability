#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include "Keyboard.h"
#include "Mouse.h"

class Window
{
private:
	GLFWwindow* m_handle;
	int m_width;
	int m_height;
public:
	Window(int width, int height);
	~Window();
	GLFWwindow* windowPtr();
	int width() const;
	int height() const;
	void setTitle(const char* title);
	void swapBuffers();
	void pollEvents();
	bool shouldClose() const;
};

Window::Window(int width, int height) : m_width(width), m_height(height)
{
	if (!glfwInit())
		throw std::runtime_error("GLFW initialization failed");
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

	m_handle = glfwCreateWindow(width, height, "Van der Pol stability", nullptr, nullptr);
	if (!m_handle)
		throw std::runtime_error("Window creation failure");
	glfwMakeContextCurrent(m_handle);
	glfwSetWindowUserPointer(m_handle, this);

	if (glewInit() != GLEW_OK)
	{
		throw std::runtime_error("GLEW initialization failed");
		glfwDestroyWindow(m_handle);
		glfwTerminate();
	}

	glfwSwapInterval(1);
	glfwSetKeyCallback(m_handle, Keyboard::keyCallback);
	glfwSetMouseButtonCallback(m_handle, Mouse::mouseButtonCallback);
	glfwSetCursorPosCallback(m_handle, Mouse::cursorPosCallback);
	glfwSetScrollCallback(m_handle, Mouse::scrollCallback);
}
Window::~Window()
{
	glfwDestroyWindow(m_handle);
	glfwTerminate();
}
inline GLFWwindow* Window::windowPtr()
{
	return m_handle; 
}
inline int Window::width() const
{
	return m_width;
}
inline int Window::height() const
{
	return m_height;
}
inline void Window::setTitle(const char* title)
{
	glfwSetWindowTitle(m_handle, title);
}
inline void Window::swapBuffers()
{
	glfwSwapBuffers(m_handle);
}
inline void Window::pollEvents()
{
	glfwPollEvents();
}
inline bool Window::shouldClose() const
{
	return glfwWindowShouldClose(m_handle);
}

