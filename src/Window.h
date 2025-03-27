#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdexcept>

struct Vec2
{
	double x, y;
	Vec2(double x_, double y_) : x(x_), y(y_) {}
	Vec2(int x_, int y_) : x(x_), y(y_) {}
};

class Window
{
private:
	GLFWwindow* m_handle;
	int m_width;
	int m_height;
	Vec2 m_cursorPos;
	double m_delta = 0.1;
	float m_param = 0.1f;
	int m_sys = 1;
	bool m_dragMode = false;
	bool m_mouseDragging = false;
public:
	Window(int width, int height);
	~Window();
	GLFWwindow* windowPtr() { return m_handle; }
	Vec2 cursorPos() const;
	int width() const;
	int height() const;
	float param() const;
	int sys() const;
private:
	void setCallbacks();

	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

	static void mouseMoveCallback(GLFWwindow* window, double x, double y);

	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

	void printInstructions();
};

Window::Window(int width, int height) : m_width(width), m_height(height), m_cursorPos(width / 2.0, height / 2.0)
{
	if (!glfwInit())
		throw std::runtime_error("GLFW initialization failed");
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	m_handle = glfwCreateWindow(width, height, "Stability", nullptr, nullptr);
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
	setCallbacks();
	printInstructions();
}
Window::~Window()
{
	glfwDestroyWindow(m_handle);
	glfwTerminate();
}
inline Vec2 Window::cursorPos() const
{
	return m_cursorPos;
}
inline int Window::width() const
{
	return m_width;
}
inline int Window::height() const
{
	return m_height;
}
inline float Window::param() const
{
	return m_param;
}
inline int Window::sys() const
{
	return m_sys;
}
void Window::setCallbacks()
{
	glfwSetKeyCallback(m_handle, keyCallback);
	glfwSetCursorPosCallback(m_handle, mouseMoveCallback);
	glfwSetMouseButtonCallback(m_handle, mouseButtonCallback);
}
void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	auto thisWindow = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_A)
			thisWindow->m_dragMode = !thisWindow->m_dragMode;
		if (key == GLFW_KEY_ESCAPE)
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		if (key == GLFW_KEY_0) thisWindow->m_cursorPos.x -= thisWindow->m_sys = 0;
		if (key == GLFW_KEY_1) thisWindow->m_cursorPos.x += thisWindow->m_sys = 1;
		if (key == GLFW_KEY_2) thisWindow->m_cursorPos.y -= thisWindow->m_sys = 2;
		if (key == GLFW_KEY_UP) thisWindow->m_param += thisWindow->m_delta;
		if (key == GLFW_KEY_DOWN) thisWindow->m_param -= thisWindow->m_delta;
	}
	if (action == GLFW_REPEAT)
	{
		if (key == GLFW_KEY_UP) thisWindow->m_param += thisWindow->m_delta;
		if (key == GLFW_KEY_DOWN) thisWindow->m_param -= thisWindow->m_delta;
	}
}

void Window::mouseMoveCallback(GLFWwindow* window, double x, double y)
{
	auto thisWindow = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (!thisWindow->m_dragMode || !thisWindow->m_mouseDragging)
		return;
	thisWindow->m_cursorPos.x = x;
	thisWindow->m_cursorPos.y = thisWindow->m_height - y;
}

void Window::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	auto thisWindow = static_cast<Window*>(glfwGetWindowUserPointer(window));
	if (button == GLFW_MOUSE_BUTTON_LEFT)
	{
		if (action == GLFW_PRESS)
		{
			thisWindow->m_mouseDragging = true;
			if (thisWindow->m_dragMode)
			{
				double x, y;
				glfwGetCursorPos(window, &x, &y);
				thisWindow->m_cursorPos.x = x;
				thisWindow->m_cursorPos.y = thisWindow->m_height - y;
			}
		}
		else if (action == GLFW_RELEASE)
			thisWindow->m_mouseDragging = false;
	}
}

void Window::printInstructions()
{
	printf("Stability visualizer\n");
	printf("Use number keys to select system:\n");
	printf("\t0: linear oscillator: positive stiffness\n");
	printf("\t1: linear oscillator: negative stiffness\n");
	printf("\t2: van der Pol oscillator: nonlinear damping\n");
	printf("up/down arrow keys adjust parameter value\n\n");
	printf("Choose the van der Pol (sys=2)\n");
	printf("Keep up arrow key depressed and watch the show.\n");
}