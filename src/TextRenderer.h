#pragma once

#include <array>
#include <string>

#include <glad/glad.h>

// ============================================================
// CONFIGURACIÓN
// ============================================================

constexpr const char* FONT_PATH =
    "../assets/Agdasima/Agdasima-Regular.ttf";


// ============================================================
// CARÁCTER
// ============================================================

struct Character
{
    GLuint texture = 0;
    int width = 0;
    int height = 0;
    int bearingX = 0;
    int bearingY = 0;
    unsigned int advance = 0;
};


// ============================================================
// TEXT RENDERER
// ============================================================

class TextRenderer
{
public:

    bool initialize();

    void render(
        const std::string& text,
        float x,
        float y,
        float scale,
        float screenWidth,
        float screenHeight,
        float red = 0.85f,
        float green = 0.88f,
        float blue = 0.92f
    );

    void cleanup();

private:

    std::array<Character, 128> characters{};

    GLuint textVAO = 0;
    GLuint textVBO = 0;
    GLuint textShaderProgram = 0;
};