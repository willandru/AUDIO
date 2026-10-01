#include "TextRenderer.h"

#include "Shader.h"

#include <ft2build.h>
#include FT_FREETYPE_H

#include <iostream>

// ============================================================
// INICIALIZAR FUENTE
// ============================================================

bool TextRenderer::initialize()
{
    FT_Library library;

    if (FT_Init_FreeType(&library))
    {
        std::cerr
            << "Error inicializando FreeType.\n";

        return false;
    }

    FT_Face face;

    if (FT_New_Face(
            library,
            FONT_PATH,
            0,
            &face))
    {
        std::cerr
            << "No se pudo cargar la fuente: "
            << FONT_PATH
            << '\n';

        FT_Done_FreeType(library);

        return false;
    }

    FT_Set_Pixel_Sizes(
        face,
        0,
        18
    );

    glPixelStorei(
        GL_UNPACK_ALIGNMENT,
        1
    );

    // ========================================================
    // CARGAR GLIFOS
    // ========================================================

    for (unsigned char c = 0; c < 128; ++c)
    {
        if (FT_Load_Char(
                face,
                c,
                FT_LOAD_RENDER))
        {
            continue;
        }

        GLuint texture = 0;

        glGenTextures(
            1,
            &texture
        );

        glBindTexture(
            GL_TEXTURE_2D,
            texture
        );

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R8,
            face->glyph->bitmap.width,
            face->glyph->bitmap.rows,
            0,
            GL_RED,
            GL_UNSIGNED_BYTE,
            face->glyph->bitmap.buffer
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_WRAP_S,
            GL_CLAMP_TO_EDGE
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_WRAP_T,
            GL_CLAMP_TO_EDGE
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MIN_FILTER,
            GL_LINEAR
        );

        glTexParameteri(
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_LINEAR
        );

        characters[c] = {
            texture,
            static_cast<int>(
                face->glyph->bitmap.width
            ),
            static_cast<int>(
                face->glyph->bitmap.rows
            ),
            face->glyph->bitmap_left,
            face->glyph->bitmap_top,
            static_cast<unsigned int>(
                face->glyph->advance.x
            )
        };
    }

    glPixelStorei(
        GL_UNPACK_ALIGNMENT,
        4
    );

    FT_Done_Face(face);
    FT_Done_FreeType(library);

    // ========================================================
    // VAO / VBO
    // ========================================================

    glGenVertexArrays(
        1,
        &textVAO
    );

    glGenBuffers(
        1,
        &textVBO
    );

    glBindVertexArray(textVAO);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        textVBO
    );

    glBufferData(
        GL_ARRAY_BUFFER,
        6 * 4 * sizeof(float),
        nullptr,
        GL_DYNAMIC_DRAW
    );

    glVertexAttribPointer(
        0,
        4,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        nullptr
    );

    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // ========================================================
    // SHADER
    // ========================================================

    textShaderProgram =
        Shader::createProgramFromFiles(
            "../src/shaders/text.vert",
            "../src/shaders/text.frag"
        );

    return true;
}

// ============================================================
// DIBUJAR TEXTO
// ============================================================

void TextRenderer::render(
    const std::string& text,
    float x,
    float y,
    float scale,
    float screenWidth,
    float screenHeight,
    float red,
    float green,
    float blue)
{
    glUseProgram(textShaderProgram);

    glUniform2f(
        glGetUniformLocation(
            textShaderProgram,
            "uScreenSize"
        ),
        screenWidth,
        screenHeight
    );

    glUniform3f(
        glGetUniformLocation(
            textShaderProgram,
            "uTextColor"
        ),
        red,
        green,
        blue
    );

    glActiveTexture(GL_TEXTURE0);

    glUniform1i(
        glGetUniformLocation(
            textShaderProgram,
            "uText"
        ),
        0
    );

    glBindVertexArray(textVAO);

    for (const unsigned char c : text)
    {
        if (c >= 128)
        {
            continue;
        }

        const Character& ch =
            characters[c];

        const float xpos =
            x + ch.bearingX * scale;

        const float ypos =
            y + (18 - ch.bearingY) * scale;

        const float width =
            ch.width * scale;

        const float height =
            ch.height * scale;

        if (width > 0.0f &&
            height > 0.0f)
        {
            const float vertices[6][4] =
            {
                {
                    xpos,
                    ypos,
                    0.0f,
                    0.0f
                },

                {
                    xpos,
                    ypos + height,
                    0.0f,
                    1.0f
                },

                {
                    xpos + width,
                    ypos + height,
                    1.0f,
                    1.0f
                },

                {
                    xpos,
                    ypos,
                    0.0f,
                    0.0f
                },

                {
                    xpos + width,
                    ypos + height,
                    1.0f,
                    1.0f
                },

                {
                    xpos + width,
                    ypos,
                    1.0f,
                    0.0f
                }
            };

            glBindTexture(
                GL_TEXTURE_2D,
                ch.texture
            );

            glBindBuffer(
                GL_ARRAY_BUFFER,
                textVBO
            );

            glBufferSubData(
                GL_ARRAY_BUFFER,
                0,
                sizeof(vertices),
                vertices
            );

            glDrawArrays(
                GL_TRIANGLES,
                0,
                6
            );
        }

        x +=
            (ch.advance >> 6) * scale;
    }

    glBindVertexArray(0);
}

// ============================================================
// LIMPIEZA
// ============================================================

void TextRenderer::cleanup()
{
    for (const Character& character : characters)
    {
        if (character.texture != 0)
        {
            glDeleteTextures(
                1,
                &character.texture
            );
        }
    }

    if (textVAO != 0)
    {
        glDeleteVertexArrays(
            1,
            &textVAO
        );

        textVAO = 0;
    }

    if (textVBO != 0)
    {
        glDeleteBuffers(
            1,
            &textVBO
        );

        textVBO = 0;
    }

    if (textShaderProgram != 0)
    {
        glDeleteProgram(
            textShaderProgram
        );

        textShaderProgram = 0;
    }
}