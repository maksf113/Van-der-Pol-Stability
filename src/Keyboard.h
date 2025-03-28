#pragma once
#include "Window.h"

class Keyboard
{
private:
	static bool s_keys[GLFW_KEY_LAST];
	static bool s_keysChanged[GLFW_KEY_LAST];
public:
	// key state callback
	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);
	// accessors
	static bool key(int key);
	static bool keyChanged(int key);
	static bool keyPressed(int key);
	static bool keyReleased(int key);
};

bool Keyboard::s_keys[GLFW_KEY_LAST] = { false };
bool Keyboard::s_keysChanged[GLFW_KEY_LAST] = { false };


void Keyboard::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action != GLFW_RELEASE)
	{
		if (!s_keys[key])
			s_keys[key] = true;
	}
	else
		s_keys[key] = false;
	s_keysChanged[key] = action != GLFW_REPEAT;
}
bool Keyboard::key(int key)
{
	return s_keys[key];
}
bool Keyboard::keyChanged(int key)
{
	bool temp = s_keysChanged[key];
	s_keysChanged[key] = false;
	return temp;
}
bool Keyboard::keyPressed(int key)
{
	return s_keys[key] && keyChanged(key);
}
bool Keyboard::keyReleased(int key)
{
	return !s_keys[key] && keyChanged(key);
}