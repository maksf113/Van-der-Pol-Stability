#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include "GLError.h"

class Shader
{
private:
	GLuint m_id;
	std::unordered_map<std::string, int> m_uniformLocationCache;
	int getUniformLocation(const std::string& name);
public:
	Shader() {}
	Shader(const std::string&, const std::string&);
	Shader(const std::string&, const std::string&, const std::string&);
	~Shader();
	void bind() const;
	void unbind() const;
	void setUniform(const std::string& name, int i);
	void setUniform(const std::string& name, float v0);
	void setUniform(const std::string& name, float v0, float v1);
	void setUniform(const std::string& name, float v0, float v1, float v2);
	void setUniform(const std::string& name, float v0, float v1, float v2, float v3);
};

const std::string readShaderSource(const std::string& filePath) 
{
	std::string content;
	std::ifstream fileStream(filePath, std::ios::in);
	if (fileStream.is_open() == false)
	{
		std::cout << "Cannot open file: " << filePath << std::endl;
		return content;
	}
	std::string line = "";
	while (!fileStream.eof()) 
	{
		getline(fileStream, line);
		content.append(line + "\n");
	}
	fileStream.close();
	return content;
}

void printShaderLog(GLuint shader)
{
	int length = 0;
	GL(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length));
	if (length > 0)
	{
		std::vector<char> log(length + 1); // +1 for null terminator
		GL(glGetShaderInfoLog(shader, length, nullptr, &log[0]));
		std::cout << "Shader info log: " << log.data() << std::endl;
	}
}

void printProgramLog(GLuint prog)
{
	int length = 0;
	GL(glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &length));
	if (length > 0)
	{
		std::vector<char> log(length + 1); // +1 for null terminator
		GL(glGetProgramInfoLog(prog, length, nullptr, &log[0]));
		std::cout << "Shader info log: " << log.data() << std::endl;
	}
}

Shader::Shader(const std::string& vertPath, const std::string& fragPath)
{
	const std::string vertSource = readShaderSource(vertPath);
	const std::string fragSource = readShaderSource(fragPath);
	const char* vertSrcCstr = vertSource.c_str();
	const char* fragSrcCstr = fragSource.c_str();
	GL(GLuint vs = glCreateShader(GL_VERTEX_SHADER));
	GL(GLuint fs = glCreateShader(GL_FRAGMENT_SHADER));
	GLint vertCompiled;
	GLint fragCompiled;
	GLint linked;

	GL(glShaderSource(vs, 1, &vertSrcCstr, NULL));
	GL(glCompileShader(vs));
	GL(glGetShaderiv(vs, GL_COMPILE_STATUS, &vertCompiled));
	if (vertCompiled != 1)
	{
		std::cout << "Veretx compilation failed" << std::endl;
		printShaderLog(vs);
	}

	GL(glShaderSource(fs, 1, &fragSrcCstr, NULL));
	GL(glCompileShader(fs));
	GL(glGetShaderiv(fs, GL_COMPILE_STATUS, &fragCompiled));
	if (fragCompiled != 1)
	{
		std::cout << "Fragment compilation failed" << std::endl;
		printShaderLog(fs);
	}

	GL(m_id = glCreateProgram());
	GL(glAttachShader(m_id, vs));
	GL(glAttachShader(m_id, fs));
	GL(glLinkProgram(m_id));

	GL(glGetProgramiv(m_id, GL_LINK_STATUS, &linked));
	if (linked != 1)
	{
		std::cout << "Program linking failed" << std::endl;
		printProgramLog(m_id);
	}
	GL(glDeleteShader(vs));
	GL(glDeleteShader(fs));
}

Shader::Shader(const std::string& vertPath, const std::string& geomPath, const std::string& fragPath)
{
	const std::string vertSource = readShaderSource(vertPath);
	const std::string geomSource = readShaderSource(geomPath);
	const std::string fragSource = readShaderSource(fragPath);
	const char* vertSrcCstr = vertSource.c_str();
	const char* geomSrcCstr = geomSource.c_str();
	const char* fragSrcCstr = fragSource.c_str();
	GL(GLuint vs = glCreateShader(GL_VERTEX_SHADER));
	GL(GLuint gs = glCreateShader(GL_GEOMETRY_SHADER));
	GL(GLuint fs = glCreateShader(GL_FRAGMENT_SHADER));
	GLint vertCompiled;
	GLint geomCompiled;
	GLint fragCompiled;
	GLint linked;

	GL(glShaderSource(vs, 1, &vertSrcCstr, NULL));
	GL(glCompileShader(vs));
	GL(glGetShaderiv(vs, GL_COMPILE_STATUS, &vertCompiled));
	if (vertCompiled != 1)
	{
		std::cout << "Veretx compilation failed" << std::endl;
		printShaderLog(vs);
	}

	GL(glShaderSource(gs, 1, &geomSrcCstr, NULL));
	GL(glCompileShader(gs));
	GL(glGetShaderiv(gs, GL_COMPILE_STATUS, &geomCompiled));

	if (geomCompiled != 1)
	{
		std::cout << "Geometry compilation failed" << std::endl;
		printShaderLog(gs);
	}

	GL(glShaderSource(fs, 1, &fragSrcCstr, NULL));
	GL(glCompileShader(fs));
	GL(glGetShaderiv(fs, GL_COMPILE_STATUS, &fragCompiled));
	if (fragCompiled != 1)
	{
		std::cout << "Fragment compilation failed" << std::endl;
		printShaderLog(fs);
	}

	GL(m_id = glCreateProgram());
	GL(glAttachShader(m_id, vs));
	GL(glAttachShader(m_id, gs));
	GL(glAttachShader(m_id, fs));
	GL(glLinkProgram(m_id));

	GL(glGetProgramiv(m_id, GL_LINK_STATUS, &linked));
	if (linked != 1)
	{
		std::cout << "Program linking failed" << std::endl;
		printProgramLog(m_id);
	}
	GL(glDeleteShader(vs));
	GL(glDeleteShader(gs));
	GL(glDeleteShader(fs));
}

Shader::~Shader()
{
	GL(glDeleteProgram(m_id));
}

inline void Shader::bind() const
{
	GL(glUseProgram(m_id));
}

inline void Shader::unbind() const
{
	GL(glUseProgram(0));
}

inline void Shader::setUniform(const std::string& name, int i)
{
	int location = getUniformLocation(name);
	GL(glUniform1i(location, i));
}
inline void Shader::setUniform(const std::string& name, float v0)
{
	int location = getUniformLocation(name);
	GL(glUniform1f(location, v0));
}

inline void Shader::setUniform(const std::string& name, float v0, float v1)
{
	int location = getUniformLocation(name);
	GL(glUniform2f(location, v0, v1));
}

inline void Shader::setUniform(const std::string& name, float v0, float v1, float v2)
{
	int location = getUniformLocation(name);
	GL(glUniform3f(location, v0, v1, v2));
}

inline void Shader::setUniform(const std::string& name, float v0, float v1, float v2, float v3)
{
	int location = getUniformLocation(name);
	GL(glUniform4f(location, v0, v1, v2, v3));
}

inline int Shader::getUniformLocation(const std::string& name)
{
	if (m_uniformLocationCache.find(name) != m_uniformLocationCache.end())
		return m_uniformLocationCache[name];
	GL(int location = glGetUniformLocation(m_id, name.c_str()));
	m_uniformLocationCache[name] = location;
	if (location == -1)
		std::cout << "Warning: uniform '" << name << "' does not exist!" << std::endl;
	return location;
}


