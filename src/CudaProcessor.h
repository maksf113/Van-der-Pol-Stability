#pragma once
#include "kernel.h"
#include "PBO.h"

class CudaProcessor
{
private:
	PBO m_pbo;
	int m_width;
	int m_height;
public:
	CudaProcessor(int width, int height);
	~CudaProcessor() = default;
	void process(float param);
	void bindPBO() const;
};

CudaProcessor::CudaProcessor(int width, int height) : m_width(width), m_height(height), m_pbo(width, height, GL_RGBA, GL_UNSIGNED_BYTE) {}


inline void CudaProcessor::process(float param)
{
	m_pbo.mapToCuda();
	stabilityKernelLauncher((uchar4*)m_pbo.cudaPtr(), m_width, m_height, param);
	m_pbo.unmapFromCuda();
}

inline void CudaProcessor::bindPBO() const
{
	m_pbo.bind(GL_PIXEL_UNPACK_BUFFER);
}
