#include "Renderer.h"

#include "Shader.h"


Point screenToNDC(
    float x,
    float y,
    float width,
    float height)
{
    return {
        2.0f * x / width - 1.0f,
        1.0f - 2.0f * y / height
    };
}


void initializeLineRenderer(
    LineRenderer& renderer)
{
    const char* vertexSource = R"(
        #version 330 core

        layout (location = 0) in vec2 aPosition;

        void main()
        {
            gl_Position = vec4(
                aPosition,
                0.0,
                1.0
            );
        }
    )";

    const char* fragmentSource = R"(
        #version 330 core

        uniform vec3 uColor;

        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(
                uColor,
                1.0
            );
        }
    )";

    renderer.shader =
        Shader::createProgram(
            vertexSource,
            fragmentSource
        );

    renderer.colorLocation =
        glGetUniformLocation(
            renderer.shader,
            "uColor"
        );

    glGenVertexArrays(
        1,
        &renderer.vao
    );

    glGenBuffers(
        1,
        &renderer.vbo
    );

    glBindVertexArray(
        renderer.vao
    );

    glBindBuffer(
        GL_ARRAY_BUFFER,
        renderer.vbo
    );

    glBufferData(
        GL_ARRAY_BUFFER,
        1024 * 1024,
        nullptr,
        GL_DYNAMIC_DRAW
    );

    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        2 * sizeof(float),
        nullptr
    );

    glEnableVertexAttribArray(0);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );

    glBindVertexArray(0);
}


void addLine(
    LineRenderer& renderer,
    float x1,
    float y1,
    float x2,
    float y2,
    float screenWidth,
    float screenHeight)
{
    const Point start =
        screenToNDC(
            x1,
            y1,
            screenWidth,
            screenHeight
        );

    const Point end =
        screenToNDC(
            x2,
            y2,
            screenWidth,
            screenHeight
        );

    renderer.vertices.push_back(start.x);
    renderer.vertices.push_back(start.y);

    renderer.vertices.push_back(end.x);
    renderer.vertices.push_back(end.y);
}


void addRectangle(
    LineRenderer& renderer,
    const Rectangle& rect,
    float screenWidth,
    float screenHeight)
{
    addLine(
        renderer,
        rect.x,
        rect.y,
        rect.x + rect.width,
        rect.y,
        screenWidth,
        screenHeight
    );

    addLine(
        renderer,
        rect.x + rect.width,
        rect.y,
        rect.x + rect.width,
        rect.y + rect.height,
        screenWidth,
        screenHeight
    );

    addLine(
        renderer,
        rect.x + rect.width,
        rect.y + rect.height,
        rect.x,
        rect.y + rect.height,
        screenWidth,
        screenHeight
    );

    addLine(
        renderer,
        rect.x,
        rect.y + rect.height,
        rect.x,
        rect.y,
        screenWidth,
        screenHeight
    );
}


void drawLines(
    LineRenderer& renderer)
{
    drawLines(
        renderer,
        0.18f,
        0.22f,
        0.27f
    );
}


void drawLines(
    LineRenderer& renderer,
    float red,
    float green,
    float blue)
{
    if (renderer.vertices.empty())
        return;

    glUseProgram(renderer.shader);

    glUniform3f(
        renderer.colorLocation,
        red,
        green,
        blue
    );

    glBindVertexArray(
        renderer.vao
    );

    glBindBuffer(
        GL_ARRAY_BUFFER,
        renderer.vbo
    );

    glBufferData(
        GL_ARRAY_BUFFER,
        renderer.vertices.size() * sizeof(float),
        renderer.vertices.data(),
        GL_DYNAMIC_DRAW
    );

    glDrawArrays(
        GL_LINES,
        0,
        static_cast<GLsizei>(
            renderer.vertices.size() / 2
        )
    );

    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );

    glBindVertexArray(0);

    renderer.vertices.clear();
}


void addPlotGrid(
    LineRenderer& renderer,
    const Rectangle& plot,
    int horizontalDivisions,
    int verticalDivisions,
    float screenWidth,
    float screenHeight)
{
    for (int i = 0;
         i <= horizontalDivisions;
         ++i)
    {
        const float fraction =
            static_cast<float>(i) /
            static_cast<float>(horizontalDivisions);

        const float y =
            plot.y +
            fraction * plot.height;

        addLine(
            renderer,
            plot.x,
            y,
            plot.x + plot.width,
            y,
            screenWidth,
            screenHeight
        );
    }

    for (int i = 0;
         i <= verticalDivisions;
         ++i)
    {
        const float fraction =
            static_cast<float>(i) /
            static_cast<float>(verticalDivisions);

        const float x =
            plot.x +
            fraction * plot.width;

        addLine(
            renderer,
            x,
            plot.y,
            x,
            plot.y + plot.height,
            screenWidth,
            screenHeight
        );
    }
}


void addAxisTicks(
    LineRenderer& renderer,
    const Rectangle& plot,
    int horizontalDivisions,
    int verticalDivisions,
    float screenWidth,
    float screenHeight)
{
    constexpr float TICK_LENGTH = 7.0f;
    constexpr int SUBDIVISIONS = 5;

    for (int i = 0;
         i <= horizontalDivisions;
         ++i)
    {
        const float fraction =
            static_cast<float>(i) /
            static_cast<float>(horizontalDivisions);

        const float x =
            plot.x +
            fraction * plot.width;

        addLine(
            renderer,
            x,
            plot.y + plot.height,
            x,
            plot.y + plot.height + TICK_LENGTH,
            screenWidth,
            screenHeight
        );

        if (i < horizontalDivisions)
        {
            for (int j = 1;
                 j < SUBDIVISIONS;
                 ++j)
            {
                const float minorFraction =
                    static_cast<float>(j) /
                    static_cast<float>(SUBDIVISIONS);

                const float minorX =
                    x +
                    minorFraction *
                    (
                        plot.width /
                        static_cast<float>(
                            horizontalDivisions
                        )
                    );

                addLine(
                    renderer,
                    minorX,
                    plot.y + plot.height,
                    minorX,
                    plot.y + plot.height +
                        TICK_LENGTH * 0.5f,
                    screenWidth,
                    screenHeight
                );
            }
        }
    }

    for (int i = 0;
         i <= verticalDivisions;
         ++i)
    {
        const float fraction =
            static_cast<float>(i) /
            static_cast<float>(verticalDivisions);

        const float y =
            plot.y +
            fraction * plot.height;

        addLine(
            renderer,
            plot.x - TICK_LENGTH,
            y,
            plot.x,
            y,
            screenWidth,
            screenHeight
        );

        if (i < verticalDivisions)
        {
            for (int j = 1;
                 j < SUBDIVISIONS;
                 ++j)
            {
                const float minorFraction =
                    static_cast<float>(j) /
                    static_cast<float>(SUBDIVISIONS);

                const float minorY =
                    y +
                    minorFraction *
                    (
                        plot.height /
                        static_cast<float>(
                            verticalDivisions
                        )
                    );

                addLine(
                    renderer,
                    plot.x - TICK_LENGTH * 0.5f,
                    minorY,
                    plot.x,
                    minorY,
                    screenWidth,
                    screenHeight
                );
            }
        }
    }
}


void createSpectrogramQuad(
    GLuint& vao,
    GLuint& vbo)
{
    glGenVertexArrays(
        1,
        &vao
    );

    glGenBuffers(
        1,
        &vbo
    );

    glBindVertexArray(vao);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        vbo
    );

    const float vertices[] =
    {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,

        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(vertices),
        vertices,
        GL_STATIC_DRAW
    );

    glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        nullptr
    );

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(
        1,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        reinterpret_cast<void*>(
            2 * sizeof(float)
        )
    );

    glEnableVertexAttribArray(1);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );

    glBindVertexArray(0);
}


void updateSpectrogramQuad(
    GLuint vbo,
    const Rectangle& plot,
    float screenWidth,
    float screenHeight)
{
    const Point bottomLeft =
        screenToNDC(
            plot.x,
            plot.y + plot.height,
            screenWidth,
            screenHeight
        );

    const Point bottomRight =
        screenToNDC(
            plot.x + plot.width,
            plot.y + plot.height,
            screenWidth,
            screenHeight
        );

    const Point topRight =
        screenToNDC(
            plot.x + plot.width,
            plot.y,
            screenWidth,
            screenHeight
        );

    const Point topLeft =
        screenToNDC(
            plot.x,
            plot.y,
            screenWidth,
            screenHeight
        );

    const float vertices[] =
    {
        bottomLeft.x,
        bottomLeft.y,
        0.0f,
        0.0f,

        bottomRight.x,
        bottomRight.y,
        1.0f,
        0.0f,

        topRight.x,
        topRight.y,
        1.0f,
        1.0f,

        bottomLeft.x,
        bottomLeft.y,
        0.0f,
        0.0f,

        topRight.x,
        topRight.y,
        1.0f,
        1.0f,

        topLeft.x,
        topLeft.y,
        0.0f,
        1.0f
    };

    glBindBuffer(
        GL_ARRAY_BUFFER,
        vbo
    );

    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        sizeof(vertices),
        vertices
    );

    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );
}