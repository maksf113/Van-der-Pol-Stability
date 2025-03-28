#pragma once
#include <GL/glew.h>
#include <vector>
#include "GLError.h"

class EBO
{
private:
	unsigned int m_id;
	unsigned int m_count;
	int m_mode;
public:
	EBO(GLenum mode = GL_STATIC_DRAW);
	EBO(const std::vector<unsigned int>& indices, GLenum mode = GL_STATIC_DRAW);
	~EBO();
	void bind() const;
	void unbind() const;
	void data(const std::vector<unsigned int>& indices);
	unsigned int count() const;
};

EBO::EBO(GLenum mode) : m_mode(mode)
{
	GL(glGenBuffers(1, &m_id));
}
inline EBO::EBO(const std::vector<unsigned int>& indices, GLenum mode) : m_mode(mode), m_count(indices.size())
{
	GL(glGenBuffers(1, &m_id));
	data(indices);
}
inline EBO::~EBO()
{
	GL(glDeleteBuffers(1, &m_id));
}

inline void EBO::bind() const
{
	GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_id));
}
inline void EBO::unbind() const
{
	GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
}

inline void EBO::data(const std::vector<unsigned int>& indices)
{
	bind();
	GL(glBufferData(GL_ELEMENT_ARRAY_BUFFER, GLsizeiptr(indices.size() * sizeof(GLuint)), indices.data(), m_mode));
	unbind();
}

inline unsigned int EBO::count() const
{
	return m_count;
}
