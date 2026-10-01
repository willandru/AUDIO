#include "Shader.h"

#include <fstream>
#include <iostream>
#include <sstream>

// ============================================================
// LEER ARCHIVO
// ============================================================

std::string Shader::readFile(const std::string& path)
{
    std::ifstream file(path);

    if (!file.is_open())
    {
        std::cerr
            << "Error cargando shader: "
            << path
            << '\n';

        return {};
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    return buffer.str();
}


// ============================================================
// COMPILAR SHADER
// ============================================================

GLuint Shader::compile(
    GLenum type,
    const char* source)
{
    GLuint shader = glCreateShader(type);

    glShaderSource(
        shader,
        1,
        &source,
        nullptr
    );

    glCompileShader(shader);

    GLint success = GL_FALSE;

    glGetShaderiv(
        shader,
        GL_COMPILE_STATUS,
        &success
    );

    if (!success)
    {
        char infoLog[1024]{};

        glGetShaderInfoLog(
            shader,
            sizeof(infoLog),
            nullptr,
            infoLog
        );

        std::cerr
            << "Error compilando shader:\n"
            << infoLog
            << '\n';
    }

    return shader;
}


// ============================================================
// CREAR PROGRAMA
// ============================================================

GLuint Shader::createProgram(
    const char* vertexSource,
    const char* fragmentSource)
{
    const GLuint vertexShader =
        compile(
            GL_VERTEX_SHADER,
            vertexSource
        );

    const GLuint fragmentShader =
        compile(
            GL_FRAGMENT_SHADER,
            fragmentSource
        );

    GLuint program = glCreateProgram();

    glAttachShader(
        program,
        vertexShader
    );

    glAttachShader(
        program,
        fragmentShader
    );

    glLinkProgram(program);

    GLint success = GL_FALSE;

    glGetProgramiv(
        program,
        GL_LINK_STATUS,
        &success
    );

    if (!success)
    {
        char infoLog[1024]{};

        glGetProgramInfoLog(
            program,
            sizeof(infoLog),
            nullptr,
            infoLog
        );

        std::cerr
            << "Error enlazando programa:\n"
            << infoLog
            << '\n';
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}


// ============================================================
// CREAR PROGRAMA DESDE ARCHIVOS
// ============================================================

GLuint Shader::createProgramFromFiles(
    const std::string& vertexPath,
    const std::string& fragmentPath)
{
    const std::string vertexSource =
        readFile(vertexPath);

    const std::string fragmentSource =
        readFile(fragmentPath);

    if (vertexSource.empty() || fragmentSource.empty())
    {
        return 0;
    }

    return createProgram(
        vertexSource.c_str(),
        fragmentSource.c_str()
    );
}