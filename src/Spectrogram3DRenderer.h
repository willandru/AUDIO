#pragma once

#include <glad/glad.h>

class Spectrogram3DRenderer
{
public:
    Spectrogram3DRenderer();
    ~Spectrogram3DRenderer();

    Spectrogram3DRenderer(
        const Spectrogram3DRenderer&
    ) = delete;

    Spectrogram3DRenderer& operator=(
        const Spectrogram3DRenderer&
    ) = delete;

    bool initialize(
        int width,
        int height
    );

    void update(
        const unsigned char* data,
        int width,
        int height,
        int currentColumn
    );

    void render(
        GLuint shader
    ) const;

    void destroy();

private:
    struct Vertex
    {
        float x;
        float y;
        float z;
    };

    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;

    int width = 0;
    int height = 0;

    int meshWidth = 0;
    int meshHeight = 0;

    int vertexCount = 0;
    int indexCount = 0;

    void createMesh();
};