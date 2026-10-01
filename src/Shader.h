#pragma once

#include <glad/glad.h>

#include <string>

class Shader
{
public:

    static GLuint compile(
        GLenum type,
        const char* source
    );

    static GLuint createProgram(
        const char* vertexSource,
        const char* fragmentSource
    );

    static GLuint createProgramFromFiles(
        const std::string& vertexPath,
        const std::string& fragmentPath
    );

private:

    static std::string readFile(
        const std::string& path
    );
};