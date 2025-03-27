#pragma once
#include <GL/glew.h>
#include <vector>
#include "GLError.h"

class VBO
{
private:
	GLuint m_id;
	GLenum m_target;
	GLenum m_mode;
public:
	VBO(GLenum target = GL_ARRAY_BUFFER, GLenum mode = GL_STATIC_DRAW);
	template<typename T>
	VBO(const std::vector<T>& vertices, GLenum target = GL_ARRAY_BUFFER, GLenum mode = GL_STATIC_DRAW);
	~VBO();
	void bind() const;
	void unbind() const;
	template<typename T>
	void data(const std::vector<T>& vertices);
};

VBO::VBO(GLenum target, GLenum mode) : m_target(target), m_mode(mode)
{
	GL(glGenBuffers(1, &m_id));
}

template<typename T>
inline VBO::VBO(const std::vector<T>& vertices, GLenum target, GLenum mode) : m_target(target), m_mode(mode)
{
	GL(glGenBuffers(1, &m_id));
	VBO::data(vertices);
}
inline VBO::~VBO()
{
	GL(glDeleteBuffers(1, &m_id));
}

inline void VBO::bind() const
{
	GL(glBindBuffer(m_target, m_id));
}
inline void VBO::unbind() const
{
	GL(glBindBuffer(m_target, 0));
}

template<typename T>
inline void VBO::data(const std::vector<T>& vertices)
{
	bind();
	GL(glBufferData(m_target, GLsizeiptr(vertices.size() * sizeof(T)), vertices.data(), m_mode));
	unbind();
}