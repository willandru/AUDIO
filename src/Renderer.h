#pragma once

#include <glad/glad.h>

#include <vector>


struct Point
{
    float x;
    float y;
};


struct Rectangle
{
    float x;
    float y;
    float width;
    float height;
};


struct LineRenderer
{
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint shader = 0;
    GLint colorLocation = -1;

    std::vector<float> vertices;
};


Point screenToNDC(
    float x,
    float y,
    float width,
    float height
);


void initializeLineRenderer(
    LineRenderer& renderer
);


void addLine(
    LineRenderer& renderer,
    float x1,
    float y1,
    float x2,
    float y2,
    float screenWidth,
    float screenHeight
);


void addRectangle(
    LineRenderer& renderer,
    const Rectangle& rect,
    float screenWidth,
    float screenHeight
);


void drawLines(
    LineRenderer& renderer
);


void drawLines(
    LineRenderer& renderer,
    float red,
    float green,
    float blue
);


void addPlotGrid(
    LineRenderer& renderer,
    const Rectangle& plot,
    int horizontalDivisions,
    int verticalDivisions,
    float screenWidth,
    float screenHeight
);


void addAxisTicks(
    LineRenderer& renderer,
    const Rectangle& plot,
    int horizontalDivisions,
    int verticalDivisions,
    float screenWidth,
    float screenHeight
);


void createSpectrogramQuad(
    GLuint& vao,
    GLuint& vbo
);


void updateSpectrogramQuad(
    GLuint vbo,
    const Rectangle& plot,
    float screenWidth,
    float screenHeight
);