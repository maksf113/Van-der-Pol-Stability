#pragma once
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "GLError.h"
#include "kernel.h"

class PBO
{
private:
    GLuint m_id;
    GLsizei m_width;
    GLsizei m_height;
    GLenum m_format;
    GLenum m_type;
    cudaGraphicsResource* m_cudaResource = nullptr;
    void* m_cudaPtr = nullptr;
    size_t m_size; // in bytes

public:
    PBO(GLsizei width, GLsizei height, GLenum format, GLenum type);
    PBO(const PBO&) = delete;
    PBO(PBO&&) = delete;
    PBO& operator=(const PBO&) = delete;
    void bind(GLenum target) const;
    void unbind(GLenum target) const;
    GLuint id() const;
    void* cudaPtr() const;
    void mapToCuda();
    void unmapFromCuda();
    ~PBO();
};
PBO::PBO(GLsizei width, GLsizei height, GLenum format, GLenum type) : m_width(width), m_height(height), m_format(format), m_type(type)
{
    int channels;
    switch (m_format)
    {
    case GL_RGBA:
        channels = 4;
        break;
    case GL_RGB:
        channels = 3;
        break;
    case GL_RED:
        channels = 1;
        break;
    default:
        channels = 0;
    }
    if (channels == 0)
        throw std::runtime_error("Unsupported PBO format");
    int bytesPerChannel;
    switch (m_type)
    {
    case GL_UNSIGNED_BYTE:
        bytesPerChannel = 1;
        break;
    case GL_UNSIGNED_SHORT:
        bytesPerChannel = 2;
        break;
    case GL_UNSIGNED_INT:
        bytesPerChannel = 4;
        break;
    case GL_HALF_FLOAT:
        bytesPerChannel = 2;
        break;
    case GL_FLOAT:
        bytesPerChannel = 4;
        break;
    default:
        bytesPerChannel = 0;
    }
    if (bytesPerChannel == 0)
        throw std::runtime_error("Unsupported PBO data type");
    m_size = m_width * m_height * channels * bytesPerChannel;
    GL(glGenBuffers(1, &m_id));
    GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_id));
    GL(glBufferData(GL_PIXEL_UNPACK_BUFFER, m_size, nullptr, GL_STREAM_DRAW));
    CUDA(cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_id, cudaGraphicsRegisterFlagsWriteDiscard));
}
void PBO::bind(GLenum target) const
{
    GL(glBindBuffer(target, m_id));
}
void PBO::unbind(GLenum target) const
{
    GL(glBindBuffer(target, 0));
}
GLuint PBO::id() const
{
    return m_id;
}
void* PBO::cudaPtr() const
{
    return m_cudaPtr;
}
void PBO::mapToCuda()
{
    CUDA(cudaGraphicsMapResources(1, &m_cudaResource, 0));
    size_t mappedSize;
    CUDA(cudaGraphicsResourceGetMappedPointer(&m_cudaPtr, &mappedSize, m_cudaResource));
    if (mappedSize != m_size)
        throw std::runtime_error("Mapped PBO size mismatch");
}
void PBO::unmapFromCuda()
{
    if (m_cudaPtr == nullptr)
        throw std::runtime_error("Cannot unmap resource: PBO not mapped to CUDA");
    CUDA(cudaGraphicsUnmapResources(1, &m_cudaResource, 0));
    m_cudaPtr = nullptr;
}
PBO::~PBO()
{
    if (m_cudaPtr)
        unmapFromCuda();
    CUDA(cudaGraphicsUnregisterResource(m_cudaResource));
    GL(glDeleteBuffers(1, &m_id))
}