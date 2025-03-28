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

class Renderer
{
private:
	std::unique_ptr<VAO> m_vao;
	std::unique_ptr<VBO> m_vbo;
	std::unique_ptr<EBO> m_ebo;
	Shader m_shader;
	Texture m_texture;
public:
	Renderer(int width, int height) : m_texture(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE), m_shader("shaders/texture.vert", "shaders/texture.frag")
	{
		VBL layout;
		layout.push<float>(2); // coords
		layout.push<float>(2); // tex coords
		std::vector<float> vertices
		{
			// Coords    // texCoords
			 1.0f, -1.0f,  1.0f, 0.0f,
			-1.0f, -1.0f,  0.0f, 0.0f,
			-1.0f,  1.0f,  0.0f, 1.0f,
			 1.0f,  1.0f,  1.0f, 1.0f,
		};
		m_vbo = std::make_unique<VBO>(vertices);
		std::vector<unsigned int> indices  { 0, 1, 2, 2, 0, 3 };
		m_ebo = std::make_unique<EBO>(indices);
		m_vao = std::make_unique<VAO>();
		m_vao->addVertexBuffer(*m_vbo, layout);
		m_vao->addIndexBuffer(*m_ebo);
		m_shader.bind();
	}
	~Renderer() {}


	void render()
	{
		GL(glClearColor(0.3, 0.0, 0.0, 1.0));
		GL(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
		m_vao->bind();
		m_texture.bind(0);
		m_texture.data(nullptr);
		m_shader.bind();
		m_shader.setUniform("u_tex", 0);
		GL(glDrawElements(GL_TRIANGLE_STRIP, m_ebo->count(), GL_UNSIGNED_INT, nullptr));
	}
};