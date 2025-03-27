#pragma once
#include "VBO.h"
#include "EBO.h"
#include "VBL.h"

class VAO
{
private:
	GLuint m_id;
public:
	VAO();
	~VAO();
	void bind() const;
	void unbind() const;
	void addVertexBuffer(const VBO& vb, const VBL& layout);
	void addIndexBuffer(const EBO& eb);
};

VAO::VAO()
{
	GL(glGenVertexArrays(1, &m_id));
}
VAO::~VAO()
{
	GL(glDeleteVertexArrays(1, &m_id));
}

inline void VAO::bind() const
{
	GL(glBindVertexArray(m_id));
}

inline void VAO::unbind() const
{
	GL(glBindVertexArray(0));
}

inline void VAO::addVertexBuffer(const VBO& vb, const VBL& layout)
{
	bind();
	vb.bind();
	const auto& elements = layout.getElements();
	for (int i = 0; i < elements.size(); i++)
	{
		const auto& element = elements[i];
		GL(glEnableVertexAttribArray(i));
		GL(glVertexAttribPointer(i, element.count, element.type, 
			element.normalized ? GL_TRUE : GL_FALSE, layout.getStride(), 
			(const void*)element.offset));
	}
	unbind();
}

inline void VAO::addIndexBuffer(const EBO& ebo)
{
	bind();
	ebo.bind();
	unbind();
}