#pragma once
#include "Window.h"

class Mouse
{
private:
	static bool s_buttons[];
	static bool s_buttonsChanged[];
	static double s_x;
	static double s_y;
	static double s_last_x;
	static double s_last_y;
	static double s_dx;
	static double s_dy;
	static double s_scroll_dx;
	static double s_scroll_dy;
	static bool s_firstMouse;
public:
	static void cursorPosCallback(GLFWwindow* window, double x, double y);
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mode);
	static void scrollCallback(GLFWwindow* window, double dx, double dy);

	static double x();
	static double y();

	static double dx();
	static double dy();

	static double scroll_dx();
	static double scroll_dy();

	static bool button(int button);
	static bool buttonChanged(int button);
	static bool buttonPressed(int button);
	static bool buttonReleased(int button);
};

bool Mouse::s_buttons[GLFW_MOUSE_BUTTON_LAST] = { false };
bool Mouse::s_buttonsChanged[GLFW_MOUSE_BUTTON_LAST] = { false };

double Mouse::s_x = 0.0;
double Mouse::s_y = 0.0;
double Mouse::s_last_x = 0.0;
double Mouse::s_last_y = 0.0;
double Mouse::s_dx = 0.0;
double Mouse::s_dy = 0.0;
double Mouse::s_scroll_dx = 0.0;
double Mouse::s_scroll_dy = 0.0;
bool Mouse::s_firstMouse = true;

void Mouse::cursorPosCallback(GLFWwindow* window, double x, double y)
{
	s_x = x;
	s_y = y;
	if (s_firstMouse)
	{
		s_last_x = x;
		s_last_y = y;
		s_firstMouse = false;
	}
	s_dx = x - s_last_x;
	s_dy = -(y - s_last_y); // inverted
	s_last_x = x;
	s_last_y = y;
}
void Mouse::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (action != GLFW_RELEASE)
	{
		if (!s_buttons[button])
			s_buttons[button] = true;
	}
	else
		s_buttons[button] = false;
	s_buttonsChanged[button] = action != GLFW_REPEAT;
}
void Mouse::scrollCallback(GLFWwindow* window, double dx, double dy)
{
	s_scroll_dx = dx;
	s_scroll_dx = dy;
}
double Mouse::x()
{
	return s_x;
}
double Mouse::y()
{
	return s_y;
}

double Mouse::dx()
{
	return s_dx;
}
double Mouse::dy()
{
	return s_dy;
}

double Mouse::scroll_dx()
{
	return s_dx;
}
double Mouse::scroll_dy()
{
	return s_scroll_dy;
}

bool Mouse::button(int button)
{
	return 0;
}
bool Mouse::buttonChanged(int button)
{
	return 0;
}
bool Mouse::buttonPressed(int button)
{
	return 0;
}
bool Mouse::buttonReleased(int button)
{
	return 0;
}