#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <memory>
#include "GLError.h"
#include "VAO.h"
#include "VBO.h"
#include "EBO.h"
#include "Shader.h"
#include "Window.h"
#include "Texture.h"
#include "PBO.h"
#include "kernel.h"

class Renderer
{
private:
	Window m_window;
	std::unique_ptr<VAO> m_vao;
	std::unique_ptr<VBO> m_vbo;
	std::unique_ptr<EBO> m_ebo;
	std::unique_ptr<Shader> m_shader;
	std::vector<float> m_vertices;
	std::vector<unsigned int> m_indices;
	PBO m_pbo;
	Texture m_texture;
public:
	Renderer(int width, int height) : m_window(width, height), m_pbo(width, height, GL_RGBA, GL_UNSIGNED_BYTE), m_texture(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE)
	{
		VBL layout;
		layout.push<float>(2); // coords
		layout.push<float>(2); // tex coords
		m_vertices =
		{
			// Coords    // texCoords
			 1.0f, -1.0f,  1.0f, 0.0f,
			-1.0f, -1.0f,  0.0f, 0.0f,
			-1.0f,  1.0f,  0.0f, 1.0f,
			 1.0f,  1.0f,  1.0f, 1.0f,
		};
		m_vbo = std::make_unique<VBO>(m_vertices);
		m_indices = { 0, 1, 2, 2, 0, 3 };
		m_ebo = std::make_unique<EBO>(m_indices);
		m_vao = std::make_unique<VAO>();
		m_vao->addVertexBuffer(*m_vbo, layout);
		m_vao->addIndexBuffer(*m_ebo);
		m_shader = std::make_unique<Shader>("shaders/texture.vert",
			"shaders/texture.frag");
		m_shader->bind();
	}
	~Renderer() {}
	GLFWwindow* windowPtr() { return m_window.windowPtr(); }
	Vec2 cursorPos() { return m_window.cursorPos(); }

	void display()
	{
		GL(glClearColor(0.3, 0.0, 0.0, 1.0));
		GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
		m_vao->bind();
		m_shader->bind();
		m_shader->setUniform("u_tex", 0);
		GL(glDrawElements(GL_TRIANGLE_STRIP, m_indices.size(), GL_UNSIGNED_INT, nullptr));
	}
	void render()
	{
		while (!glfwWindowShouldClose(windowPtr()))
		{
			GL(glClear(GL_COLOR_BUFFER_BIT));
			m_pbo.bind(GL_PIXEL_UNPACK_BUFFER);
			m_texture.bind(0);
			m_texture.data(nullptr);
			m_pbo.mapToCuda();
			stabilityKernelLauncher((uchar4*)m_pbo.cudaPtr(), m_window.width(), m_window.height(), m_window.param(), m_window.sys());
			m_pbo.unmapFromCuda();
			display();
			glfwSwapBuffers(m_window.windowPtr());
			glfwPollEvents();
		}
	}
};