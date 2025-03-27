#pragma once
#include <GL/glew.h>
#include "glError.h"

class Texture
{
private:
    GLuint m_id;
    GLsizei m_width;
    GLsizei m_height;
    GLenum m_internalFormat;
    GLenum m_format;
    GLenum m_type;

public:
    Texture(GLsizei width, GLsizei height, GLenum internalFormat, GLenum format, GLenum type);
    void bind() const;
    void bind(GLint i) const;
    void unbind() const;
    void data(const void* data);
    GLsizei id() const;
    ~Texture();
};

Texture::Texture(GLsizei width, GLsizei height, GLenum internalFormat, GLenum format, GLenum type)
    : m_width(width), m_height(height), m_format(format), m_internalFormat(internalFormat), m_type(type)
{
    GL(glGenTextures(1, &m_id));
    bind();
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
}
void Texture::bind() const
{
    GL(glBindTexture(GL_TEXTURE_2D, m_id));
}
void Texture::bind(GLint i) const
{
    GL(glActiveTexture(GL_TEXTURE0 + i));
    GL(glBindTexture(GL_TEXTURE_2D, m_id));
}
void Texture::unbind() const
{
    GL(glBindTexture(GL_TEXTURE_2D, 0));
}
void Texture::data(const void* data)
{
    GL(glTexImage2D(GL_TEXTURE_2D, 0, m_internalFormat, m_width, m_height, 0, m_format, m_type, data));
}
GLsizei Texture::id() const
{
    return m_id;
}
Texture::~Texture()
{
    GL(glDeleteTextures(1, &m_id));
}