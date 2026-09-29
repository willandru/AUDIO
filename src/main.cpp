#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <portaudio.h>

#include <ft2build.h>
#include FT_FREETYPE_H

// ============================================================
// CONFIGURACIÓN
// ============================================================

constexpr int SAMPLE_RATE = 44100;
constexpr int FRAMES_PER_BUFFER = 256;
constexpr int CHANNELS = 1;

constexpr int WAVEFORM_SAMPLES = 1024;
constexpr int FFT_SIZE = 1024;
constexpr int SPECTRUM_BINS = FFT_SIZE / 2;

constexpr int SPECTROGRAM_WIDTH = 1024;
constexpr int SPECTROGRAM_HEIGHT = SPECTRUM_BINS;

constexpr float TARGET_AMPLITUDE = 0.65f;
constexpr float GAIN_ATTACK = 0.20f;
constexpr float GAIN_RELEASE = 0.02f;
constexpr float MIN_GAIN = 1.0f;
constexpr float MAX_GAIN = 50.0f;

constexpr float SPECTROGRAM_MIN_DB = -100.0f;
constexpr float SPECTROGRAM_MAX_DB = 0.0f;
constexpr float SPECTROGRAM_VISUAL_GAIN = 4.0f;

constexpr float PI = 3.14159265358979323846f;

constexpr float WAVEFORM_DURATION =
    static_cast<float>(WAVEFORM_SAMPLES) /
    static_cast<float>(SAMPLE_RATE);

constexpr float SPECTROGRAM_DURATION =
    static_cast<float>(SPECTROGRAM_WIDTH) *
    static_cast<float>(FRAMES_PER_BUFFER) /
    static_cast<float>(SAMPLE_RATE);
constexpr const char* FONT_PATH =
    "../assets/Agdasima/Agdasima-Regular.ttf";
// ============================================================
// DATOS DE AUDIO
// ============================================================

struct AudioData
{
    std::array<
        std::atomic<float>,
        WAVEFORM_SAMPLES
    > samples{};

    std::atomic<int> writeIndex{ 0 };
};


// ============================================================
// FUENTE
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

std::array<Character, 128> characters{};

GLuint textVAO = 0;
GLuint textVBO = 0;
GLuint textShaderProgram = 0;


// ============================================================
// CALLBACK DE AUDIO
// ============================================================

int audioCallback(
    const void* input,
    void*,
    unsigned long frameCount,
    const PaStreamCallbackTimeInfo*,
    PaStreamCallbackFlags,
    void* userData)
{
    auto* audioData =
        static_cast<AudioData*>(userData);

    if (input == nullptr)
    {
        return paContinue;
    }

    const auto* inputSamples =
        static_cast<const float*>(input);

    for (unsigned long i = 0;
         i < frameCount;
         ++i)
    {
        const int index =
            audioData->writeIndex.fetch_add(
                1,
                std::memory_order_relaxed
            ) % WAVEFORM_SAMPLES;

        audioData->samples[index].store(
            inputSamples[i],
            std::memory_order_relaxed
        );
    }

    return paContinue;
}


// ============================================================
// COMPILAR SHADER
// ============================================================

GLuint compileShader(
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

GLuint createProgram(
    const char* vertexSource,
    const char* fragmentSource)
{
    const GLuint vertexShader =
        compileShader(
            GL_VERTEX_SHADER,
            vertexSource
        );

    const GLuint fragmentShader =
        compileShader(
            GL_FRAGMENT_SHADER,
            fragmentSource
        );

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);

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
// FFT
// ============================================================

void fft(
    std::array<
        std::complex<float>,
        FFT_SIZE
    >& values)
{
    int j = 0;

    for (int i = 1; i < FFT_SIZE; ++i)
    {
        int bit = FFT_SIZE >> 1;

        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j)
        {
            std::swap(values[i], values[j]);
        }
    }

    for (int length = 2;
         length <= FFT_SIZE;
         length <<= 1)
    {
        const float angle =
            -2.0f * PI /
            static_cast<float>(length);

        const std::complex<float> wlen(
            std::cos(angle),
            std::sin(angle)
        );

        for (int i = 0; i < FFT_SIZE; i += length)
        {
            std::complex<float> w(1.0f, 0.0f);

            const int halfLength = length / 2;

            for (int j = 0; j < halfLength; ++j)
            {
                const auto u = values[i + j];

                const auto v =
                    values[i + j + halfLength] * w;

                values[i + j] = u + v;

                values[i + j + halfLength] = u - v;

                w *= wlen;
            }
        }
    }
}


// ============================================================
// SHADERS DE LÍNEAS
// ============================================================

const char* lineVertexShader = R"(
    #version 330 core

    layout (location = 0) in vec2 aPosition;

    void main()
    {
        gl_Position = vec4(aPosition, 0.0, 1.0);
    }
)";

const char* lineFragmentShader = R"(
    #version 330 core

    uniform vec3 uColor;

    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(uColor, 1.0);
    }
)";


// ============================================================
// SHADERS DEL ESPECTROGRAMA
// ============================================================

const char* spectrumVertexShader = R"(
    #version 330 core

    layout (location = 0) in vec2 aPosition;
    layout (location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    void main()
    {
        gl_Position = vec4(aPosition, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
)";

const char* spectrumFragmentShader = R"(
    #version 330 core

    in vec2 TexCoord;

    uniform sampler2D uSpectrum;
    uniform float uColumn;

    out vec4 FragColor;

    void main()
    {
        float normalizedColumn = uColumn / 1024.0;

        float x = fract(
            TexCoord.x + normalizedColumn
        );

        float value = texture(
            uSpectrum,
            vec2(x, TexCoord.y)
        ).r;

        float enhanced = pow(value, 0.55);

        vec3 color;

        if (enhanced < 0.25)
        {
            float t = enhanced / 0.25;

            color = mix(
                vec3(0.0, 0.0, 0.015),
                vec3(0.0, 0.15, 0.8),
                t
            );
        }
        else if (enhanced < 0.50)
        {
            float t = (enhanced - 0.25) / 0.25;

            color = mix(
                vec3(0.0, 0.15, 0.8),
                vec3(0.0, 0.9, 1.0),
                t
            );
        }
        else if (enhanced < 0.72)
        {
            float t = (enhanced - 0.50) / 0.22;

            color = mix(
                vec3(0.0, 0.9, 1.0),
                vec3(1.0, 1.0, 0.0),
                t
            );
        }
        else
        {
            float t = (enhanced - 0.72) / 0.28;

            color = mix(
                vec3(1.0, 1.0, 0.0),
                vec3(1.0, 0.0, 0.0),
                t
            );
        }

        FragColor = vec4(color, 1.0);
    }
)";


// ============================================================
// SHADERS DE TEXTO
// ============================================================

const char* textVertexShader = R"(
    #version 330 core

    layout (location = 0) in vec4 vertex;

    uniform vec2 uScreenSize;

    out vec2 TexCoord;

    void main()
    {
        vec2 position = vertex.xy;

        vec2 normalized = vec2(
            position.x / uScreenSize.x * 2.0 - 1.0,
            1.0 - position.y / uScreenSize.y * 2.0
        );

        gl_Position = vec4(normalized, 0.0, 1.0);

        TexCoord = vertex.zw;
    }
)";

const char* textFragmentShader = R"(
    #version 330 core

    in vec2 TexCoord;

    uniform sampler2D uText;
    uniform vec3 uTextColor;

    out vec4 FragColor;

    void main()
    {
        float alpha = texture(uText, TexCoord).r;

        FragColor = vec4(
            uTextColor,
            alpha
        );
    }
)";


// ============================================================
// INICIALIZAR FUENTE
// ============================================================

bool initializeFont()
{
    FT_Library library;

    if (FT_Init_FreeType(&library))
    {
        std::cerr << "Error inicializando FreeType.\n";
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

    FT_Set_Pixel_Sizes(face, 0, 18);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

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

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

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
            static_cast<int>(face->glyph->bitmap.width),
            static_cast<int>(face->glyph->bitmap.rows),
            face->glyph->bitmap_left,
            face->glyph->bitmap_top,
            static_cast<unsigned int>(face->glyph->advance.x)
        };
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

    FT_Done_Face(face);
    FT_Done_FreeType(library);

    glGenVertexArrays(1, &textVAO);
    glGenBuffers(1, &textVBO);

    glBindVertexArray(textVAO);
    glBindBuffer(GL_ARRAY_BUFFER, textVBO);

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

    textShaderProgram = createProgram(
        textVertexShader,
        textFragmentShader
    );

    return true;
}


// ============================================================
// DIBUJAR TEXTO
// ============================================================

void renderText(
    const std::string& text,
    float x,
    float y,
    float scale,
    float screenWidth,
    float screenHeight,
    float red = 0.85f,
    float green = 0.88f,
    float blue = 0.92f)
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

        const Character& ch = characters[c];

        const float xpos =
            x + ch.bearingX * scale;

        const float ypos =
            y + (18 - ch.bearingY) * scale;

        const float width =
            ch.width * scale;

        const float height =
            ch.height * scale;

        if (width > 0.0f && height > 0.0f)
        {
            const float vertices[6][4] =
            {
                { xpos,          ypos,           0.0f, 0.0f },
                { xpos,          ypos + height,  0.0f, 1.0f },
                { xpos + width,  ypos + height,  1.0f, 1.0f },

                { xpos,          ypos,           0.0f, 0.0f },
                { xpos + width,  ypos + height,  1.0f, 1.0f },
                { xpos + width,  ypos,           1.0f, 0.0f }
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

            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

        x += (ch.advance >> 6) * scale;
    }

    glBindVertexArray(0);
}


// ============================================================
// GEOMETRÍA 2D
// ============================================================

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


// ============================================================
// CONVERSIÓN DE COORDENADAS
// ============================================================

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


// ============================================================
// INICIALIZAR LÍNEAS
// ============================================================

void initializeLineRenderer(LineRenderer& renderer)
{
    renderer.shader = createProgram(
        lineVertexShader,
        lineFragmentShader
    );

    renderer.colorLocation =
        glGetUniformLocation(
            renderer.shader,
            "uColor"
        );

    glGenVertexArrays(1, &renderer.vao);
    glGenBuffers(1, &renderer.vbo);

    glBindVertexArray(renderer.vao);
    glBindBuffer(GL_ARRAY_BUFFER, renderer.vbo);

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

    glBindVertexArray(0);
}


// ============================================================
// AGREGAR LÍNEA
// ============================================================

void addLine(
    LineRenderer& renderer,
    float x1,
    float y1,
    float x2,
    float y2,
    float screenWidth,
    float screenHeight)
{
    const Point a = screenToNDC(
        x1,
        y1,
        screenWidth,
        screenHeight
    );

    const Point b = screenToNDC(
        x2,
        y2,
        screenWidth,
        screenHeight
    );

    renderer.vertices.push_back(a.x);
    renderer.vertices.push_back(a.y);

    renderer.vertices.push_back(b.x);
    renderer.vertices.push_back(b.y);
}


// ============================================================
// RECTÁNGULO
// ============================================================

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


// ============================================================
// DIBUJAR LÍNEAS
// ============================================================

void drawLines(
    LineRenderer& renderer,
    float red,
    float green,
    float blue)
{
    if (renderer.vertices.empty())
    {
        return;
    }

    glUseProgram(renderer.shader);

    glUniform3f(
        renderer.colorLocation,
        red,
        green,
        blue
    );

    glBindVertexArray(renderer.vao);

    glBindBuffer(GL_ARRAY_BUFFER, renderer.vbo);

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

    glBindVertexArray(0);

    renderer.vertices.clear();
}


// ============================================================
// CUADRÍCULA Y EJES
// ============================================================

void addPlotGrid(
    LineRenderer& renderer,
    const Rectangle& plot,
    int horizontalDivisions,
    int verticalDivisions,
    float screenWidth,
    float screenHeight)
{
    for (int i = 0; i <= horizontalDivisions; ++i)
    {
        const float y =
            plot.y +
            plot.height *
            static_cast<float>(i) /
            static_cast<float>(horizontalDivisions);

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

    for (int i = 0; i <= verticalDivisions; ++i)
    {
        const float x =
            plot.x +
            plot.width *
            static_cast<float>(i) /
            static_cast<float>(verticalDivisions);

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


// ============================================================
// GRADUACIONES DE LOS EJES
// ============================================================

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

    for (int i = 0; i <= horizontalDivisions; ++i)
    {
        const float x =
            plot.x +
            plot.width *
            static_cast<float>(i) /
            static_cast<float>(horizontalDivisions);

        addLine(
            renderer,
            x,
            plot.y + plot.height,
            x,
            plot.y + plot.height + TICK_LENGTH,
            screenWidth,
            screenHeight
        );
    }

    for (int i = 0; i <= verticalDivisions; ++i)
    {
        const float y =
            plot.y +
            plot.height -
            plot.height *
            static_cast<float>(i) /
            static_cast<float>(verticalDivisions);

        addLine(
            renderer,
            plot.x - TICK_LENGTH,
            y,
            plot.x,
            y,
            screenWidth,
            screenHeight
        );
    }

    for (int i = 0; i < horizontalDivisions; ++i)
    {
        for (int j = 1; j < SUBDIVISIONS; ++j)
        {
            const float fraction =
                (
                    static_cast<float>(i) +
                    static_cast<float>(j) /
                    static_cast<float>(SUBDIVISIONS)
                ) /
                static_cast<float>(horizontalDivisions);

            const float x =
                plot.x + plot.width * fraction;

            addLine(
                renderer,
                x,
                plot.y + plot.height,
                x,
                plot.y + plot.height + TICK_LENGTH * 0.5f,
                screenWidth,
                screenHeight
            );
        }
    }

    for (int i = 0; i < verticalDivisions; ++i)
    {
        for (int j = 1; j < SUBDIVISIONS; ++j)
        {
            const float fraction =
                (
                    static_cast<float>(i) +
                    static_cast<float>(j) /
                    static_cast<float>(SUBDIVISIONS)
                ) /
                static_cast<float>(verticalDivisions);

            const float y =
                plot.y + plot.height -
                plot.height * fraction;

            addLine(
                renderer,
                plot.x - TICK_LENGTH * 0.5f,
                y,
                plot.x,
                y,
                screenWidth,
                screenHeight
            );
        }
    }
}


// ============================================================
// DIBUJAR ONDA
// ============================================================

void drawWaveform(
    GLuint vao,
    GLuint vbo,
    const std::array<float, WAVEFORM_SAMPLES>& samples,
    float gain,
    const Rectangle& plot,
    float screenWidth,
    float screenHeight)
{
    std::array<float, WAVEFORM_SAMPLES * 2> vertices{};

    for (int i = 0; i < WAVEFORM_SAMPLES; ++i)
    {
        const float x =
            plot.x +
            plot.width *
            static_cast<float>(i) /
            static_cast<float>(WAVEFORM_SAMPLES - 1);

        const float normalized =
            std::clamp(
                samples[i] * gain,
                -1.0f,
                1.0f
            );

        const float y =
            plot.y +
            plot.height * 0.5f -
            normalized * plot.height * 0.5f;

        const Point p = screenToNDC(
            x,
            y,
            screenWidth,
            screenHeight
        );

        vertices[i * 2] = p.x;
        vertices[i * 2 + 1] = p.y;
    }

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        sizeof(vertices),
        vertices.data()
    );

    glDrawArrays(
        GL_LINE_STRIP,
        0,
        WAVEFORM_SAMPLES
    );

    glBindVertexArray(0);
}


// ============================================================
// INICIALIZAR TEXTURA DEL ESPECTROGRAMA
// ============================================================

GLuint createSpectrogramTexture(
    const std::array<
        unsigned char,
        SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT
    >& data)
{
    GLuint texture = 0;

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

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

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_R8,
        SPECTROGRAM_WIDTH,
        SPECTROGRAM_HEIGHT,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        data.data()
    );

    return texture;
}


// ============================================================
// QUAD DEL ESPECTROGRAMA
// ============================================================

void createSpectrogramQuad(
    GLuint& vao,
    GLuint& vbo)
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    const float vertices[] =
    {
        // Posición          // Coordenadas de textura

        -1.0f, -1.0f,        0.0f, 0.0f,
         1.0f, -1.0f,        1.0f, 0.0f,
         1.0f,  1.0f,        1.0f, 1.0f,

        -1.0f, -1.0f,        0.0f, 0.0f,
         1.0f,  1.0f,        1.0f, 1.0f,
        -1.0f,  1.0f,        0.0f, 1.0f
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
        reinterpret_cast<void*>(2 * sizeof(float))
    );

    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}


// ============================================================
// ACTUALIZAR GEOMETRÍA DEL ESPECTROGRAMA
// ============================================================

void updateSpectrogramQuad(
    GLuint vbo,
    const Rectangle& plot,
    float screenWidth,
    float screenHeight)
{
    const Point bottomLeft = screenToNDC(
        plot.x,
        plot.y + plot.height,
        screenWidth,
        screenHeight
    );

    const Point bottomRight = screenToNDC(
        plot.x + plot.width,
        plot.y + plot.height,
        screenWidth,
        screenHeight
    );

    const Point topRight = screenToNDC(
        plot.x + plot.width,
        plot.y,
        screenWidth,
        screenHeight
    );

    const Point topLeft = screenToNDC(
        plot.x,
        plot.y,
        screenWidth,
        screenHeight
    );

    const float vertices[] =
    {
        bottomLeft.x,  bottomLeft.y,  0.0f, 0.0f,
        bottomRight.x, bottomRight.y, 1.0f, 0.0f,
        topRight.x,    topRight.y,    1.0f, 1.0f,

        bottomLeft.x,  bottomLeft.y,  0.0f, 0.0f,
        topRight.x,    topRight.y,    1.0f, 1.0f,
        topLeft.x,     topLeft.y,     0.0f, 1.0f
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        sizeof(vertices),
        vertices
    );
}


// ============================================================
// DIBUJAR ESPECTROGRAMA
// ============================================================

void drawSpectrogram(
    GLuint shader,
    GLuint vao,
    GLuint texture,
    GLint spectrumLocation,
    GLint columnLocation,
    int column)
{
    glUseProgram(shader);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    glUniform1i(spectrumLocation, 0);

    glUniform1f(
        columnLocation,
        static_cast<float>(column)
    );

    glBindVertexArray(vao);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindVertexArray(0);
}


// ============================================================
// ETIQUETAS DE LOS EJES
// ============================================================

void drawWaveformLabels(
    const Rectangle& panel,
    const Rectangle& plot,
    float screenWidth,
    float screenHeight)
{
    const float scale =
        std::clamp(screenHeight / 900.0f, 0.8f, 1.5f);

    renderText(
        "FORMA DE ONDA",
        panel.x + 30.0f,
        panel.y + 10.0f,
        scale * 1.35f,
        screenWidth,
        screenHeight,
        0.92f,
        0.95f,
        0.98f
    );

    renderText(
        "Amplitud",
        panel.x + 14.0f,
        plot.y + plot.height * 0.5f - 10.0f,
        scale,
        screenWidth,
        screenHeight
    );

    renderText(
        "Tiempo (ms)",
        plot.x + plot.width * 0.5f - 40.0f,
        plot.y + plot.height + 28.0f,
        scale,
        screenWidth,
        screenHeight
    );

    constexpr int Y_DIVISIONS = 4;

    for (int i = 0; i <= Y_DIVISIONS; ++i)
    {
        const float value =
            -1.0f +
            2.0f * static_cast<float>(i) /
            static_cast<float>(Y_DIVISIONS);

        const float y =
            plot.y + plot.height -
            plot.height * static_cast<float>(i) /
            static_cast<float>(Y_DIVISIONS);

        char label[32];

        std::snprintf(
            label,
            sizeof(label),
            "%.1f",
            value
        );

        renderText(
            label,
            plot.x - 38.0f,
            y - 9.0f,
            scale * 0.9f,
            screenWidth,
            screenHeight
        );
    }

    constexpr int X_DIVISIONS = 10;

    for (int i = 0; i <= X_DIVISIONS; ++i)
    {
        const float fraction =
            static_cast<float>(i) /
            static_cast<float>(X_DIVISIONS);

        const float x =
            plot.x + plot.width * fraction;

        const float milliseconds =
            fraction * WAVEFORM_DURATION * 1000.0f;

        char label[32];

        std::snprintf(
            label,
            sizeof(label),
            "%.1f",
            milliseconds
        );

        renderText(
            label,
            x - 12.0f,
            plot.y + plot.height + 5.0f,
            scale * 0.85f,
            screenWidth,
            screenHeight
        );
    }
}


// ============================================================
// ETIQUETAS DEL ESPECTROGRAMA
// ============================================================

void drawSpectrogramLabels(
    const Rectangle& panel,
    const Rectangle& plot,
    float screenWidth,
    float screenHeight)
{
    const float scale =
        std::clamp(screenHeight / 900.0f, 0.8f, 1.5f);

    renderText(
        "ESPECTROGRAMA",
        panel.x + 30.0f,
        panel.y + 10.0f,
        scale * 1.35f,
        screenWidth,
        screenHeight,
        0.92f,
        0.95f,
        0.98f
    );

    renderText(
        "Frecuencia (Hz)",
        panel.x + 12.0f,
        plot.y + plot.height * 0.5f - 10.0f,
        scale,
        screenWidth,
        screenHeight
    );

    renderText(
        "Tiempo (s)",
        plot.x + plot.width * 0.5f - 30.0f,
        plot.y + plot.height + 28.0f,
        scale,
        screenWidth,
        screenHeight
    );

    constexpr int Y_DIVISIONS = 5;

    for (int i = 0; i <= Y_DIVISIONS; ++i)
    {
        const float fraction =
            static_cast<float>(i) /
            static_cast<float>(Y_DIVISIONS);

        const float frequency =
            fraction * SAMPLE_RATE * 0.5f;

        const float y =
            plot.y + plot.height -
            plot.height * fraction;

        char label[32];

        if (frequency >= 1000.0f)
        {
            std::snprintf(
                label,
                sizeof(label),
                "%.1f k",
                frequency / 1000.0f
            );
        }
        else
        {
            std::snprintf(
                label,
                sizeof(label),
                "%.0f",
                frequency
            );
        }

        renderText(
            label,
            plot.x - 47.0f,
            y - 9.0f,
            scale * 0.9f,
            screenWidth,
            screenHeight
        );
    }

    constexpr int X_DIVISIONS = 10;

    for (int i = 0; i <= X_DIVISIONS; ++i)
    {
        const float fraction =
            static_cast<float>(i) /
            static_cast<float>(X_DIVISIONS);

        const float x =
            plot.x + plot.width * fraction;

        const float seconds =
            fraction * SPECTROGRAM_DURATION;

        char label[32];

        std::snprintf(
            label,
            sizeof(label),
            "%.1f",
            seconds
        );

        renderText(
            label,
            x - 10.0f,
            plot.y + plot.height + 5.0f,
            scale * 0.85f,
            screenWidth,
            screenHeight
        );
    }
}


// ============================================================
// PANEL Y ÁREA DE GRÁFICA
// ============================================================

void calculateLayout(
    int width,
    int height,
    Rectangle& waveformPanel,
    Rectangle& waveformPlot,
    Rectangle& spectrogramPanel,
    Rectangle& spectrogramPlot)
{
    const float margin = std::max(18.0f, width * 0.018f);
    const float gap = std::max(12.0f, height * 0.018f);

    const float availableHeight =
        static_cast<float>(height) -
        2.0f * margin -
        gap;

    const float waveformHeight =
        availableHeight * 0.43f;

    const float spectrogramHeight =
        availableHeight - waveformHeight;

    waveformPanel = {
        margin,
        margin,
        static_cast<float>(width) - 2.0f * margin,
        waveformHeight
    };

    spectrogramPanel = {
        margin,
        margin + waveformHeight + gap,
        static_cast<float>(width) - 2.0f * margin,
        spectrogramHeight
    };

    const float leftAxis = 72.0f;
    const float rightMargin = 24.0f;
    const float topMargin = 46.0f;
    const float bottomMargin = 52.0f;

    waveformPlot = {
        waveformPanel.x + leftAxis,
        waveformPanel.y + topMargin,
        waveformPanel.width - leftAxis - rightMargin,
        waveformPanel.height - topMargin - bottomMargin
    };

    spectrogramPlot = {
        spectrogramPanel.x + leftAxis,
        spectrogramPanel.y + topMargin,
        spectrogramPanel.width - leftAxis - rightMargin - 54.0f,
        spectrogramPanel.height - topMargin - bottomMargin
    };
}


// ============================================================
// DIBUJAR MARCOS Y CUADRÍCULAS
// ============================================================

void drawInterface(
    LineRenderer& renderer,
    const Rectangle& waveformPanel,
    const Rectangle& waveformPlot,
    const Rectangle& spectrogramPanel,
    const Rectangle& spectrogramPlot,
    float screenWidth,
    float screenHeight)
{
    renderer.vertices.clear();

    addRectangle(
        renderer,
        waveformPanel,
        screenWidth,
        screenHeight
    );

    addRectangle(
        renderer,
        spectrogramPanel,
        screenWidth,
        screenHeight
    );

    addRectangle(
        renderer,
        waveformPlot,
        screenWidth,
        screenHeight
    );

    addRectangle(
        renderer,
        spectrogramPlot,
        screenWidth,
        screenHeight
    );

    addPlotGrid(
        renderer,
        waveformPlot,
        4,
        10,
        screenWidth,
        screenHeight
    );

    addPlotGrid(
        renderer,
        spectrogramPlot,
        5,
        10,
        screenWidth,
        screenHeight
    );

    addAxisTicks(
        renderer,
        waveformPlot,
        4,
        10,
        screenWidth,
        screenHeight
    );

    addAxisTicks(
        renderer,
        spectrogramPlot,
        5,
        10,
        screenWidth,
        screenHeight
    );

    drawLines(
        renderer,
        0.18f,
        0.22f,
        0.27f
    );
}


// ============================================================
// MAIN
// ============================================================

int main()
{
    // ========================================================
    // PORTAUDIO
    // ========================================================

    PaError error = Pa_Initialize();

    if (error != paNoError)
    {
        std::cerr
            << "Error inicializando PortAudio: "
            << Pa_GetErrorText(error)
            << '\n';

        return 1;
    }

    const PaDeviceIndex inputDevice =
        Pa_GetDefaultInputDevice();

    if (inputDevice == paNoDevice)
    {
        std::cerr << "No se encontró dispositivo de entrada.\n";
        Pa_Terminate();
        return 1;
    }

    const PaDeviceInfo* deviceInfo =
        Pa_GetDeviceInfo(inputDevice);

    std::cout
        << "Dispositivo de entrada: "
        << deviceInfo->name
        << '\n'
        << "Canales: "
        << deviceInfo->maxInputChannels
        << '\n'
        << "Sample rate: "
        << deviceInfo->defaultSampleRate
        << " Hz\n";

    AudioData audioData;

    PaStream* stream = nullptr;

    PaStreamParameters inputParameters{};

    inputParameters.device = inputDevice;
    inputParameters.channelCount = CHANNELS;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency =
        deviceInfo->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;

    error = Pa_OpenStream(
        &stream,
        &inputParameters,
        nullptr,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        paNoFlag,
        audioCallback,
        &audioData
    );

    if (error != paNoError)
    {
        std::cerr
            << "Error abriendo stream: "
            << Pa_GetErrorText(error)
            << '\n';

        Pa_Terminate();
        return 1;
    }

    error = Pa_StartStream(stream);

    if (error != paNoError)
    {
        std::cerr
            << "Error iniciando stream: "
            << Pa_GetErrorText(error)
            << '\n';

        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }


    // ========================================================
    // GLFW
    // ========================================================

    if (!glfwInit())
    {
        std::cerr << "Error inicializando GLFW.\n";

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    if (monitor == nullptr)
    {
        std::cerr << "No se encontró el monitor principal.\n";

        glfwTerminate();
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }

    const GLFWvidmode* videoMode =
        glfwGetVideoMode(monitor);

    if (videoMode == nullptr)
    {
        std::cerr << "No se pudo obtener el modo de video.\n";

        glfwTerminate();
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(
        videoMode->width,
        videoMode->height,
        "AUDIO",
        monitor,
        nullptr
    );

    if (!window)
    {
        std::cerr << "Error creando ventana fullscreen.\n";

        glfwTerminate();
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);


    // ========================================================
    // GLAD
    // ========================================================

    if (!gladLoadGLLoader(
            reinterpret_cast<GLADloadproc>(
                glfwGetProcAddress)))
    {
        std::cerr << "Error inicializando GLAD.\n";

        glfwDestroyWindow(window);
        glfwTerminate();

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }


    // ========================================================
    // OPENGL
    // ========================================================

    int framebufferWidth = 0;
    int framebufferHeight = 0;

    glfwGetFramebufferSize(
        window,
        &framebufferWidth,
        &framebufferHeight
    );

    glViewport(
        0,
        0,
        framebufferWidth,
        framebufferHeight
    );

    std::cout
        << "OpenGL inicializado correctamente.\n"
        << "Version: "
        << glGetString(GL_VERSION)
        << '\n'
        << "Renderer: "
        << glGetString(GL_RENDERER)
        << '\n';


    // ========================================================
    // FUENTE
    // ========================================================

    if (!initializeFont())
    {
        glfwDestroyWindow(window);
        glfwTerminate();

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }


    // ========================================================
    // SHADERS
    // ========================================================

    const GLuint lineShaderProgram =
        createProgram(
            lineVertexShader,
            lineFragmentShader
        );

    const GLint lineColorLocation =
        glGetUniformLocation(
            lineShaderProgram,
            "uColor"
        );

    const GLuint spectrumShaderProgram =
        createProgram(
            spectrumVertexShader,
            spectrumFragmentShader
        );

    const GLint spectrumLocation =
        glGetUniformLocation(
            spectrumShaderProgram,
            "uSpectrum"
        );

    const GLint columnLocation =
        glGetUniformLocation(
            spectrumShaderProgram,
            "uColumn"
        );


    // ========================================================
    // WAVEFORM VAO / VBO
    // ========================================================

    GLuint waveformVAO = 0;
    GLuint waveformVBO = 0;

    glGenVertexArrays(1, &waveformVAO);
    glGenBuffers(1, &waveformVBO);

    glBindVertexArray(waveformVAO);
    glBindBuffer(GL_ARRAY_BUFFER, waveformVBO);

    glBufferData(
        GL_ARRAY_BUFFER,
        WAVEFORM_SAMPLES * 2 * sizeof(float),
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
    glBindVertexArray(0);


    // ========================================================
    // RENDERIZADOR DE LÍNEAS
    // ========================================================

    LineRenderer lineRenderer;
    initializeLineRenderer(lineRenderer);

    lineRenderer.shader = lineShaderProgram;

    lineRenderer.colorLocation = lineColorLocation;


    // ========================================================
    // TEXTURA DEL ESPECTROGRAMA
    // ========================================================

    std::array<
        unsigned char,
        SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT
    > spectrogramData{};

    const GLuint spectrogramTexture =
        createSpectrogramTexture(spectrogramData);


    // ========================================================
    // QUAD DEL ESPECTROGRAMA
    // ========================================================

    GLuint spectrogramVAO = 0;
    GLuint spectrogramVBO = 0;

    createSpectrogramQuad(
        spectrogramVAO,
        spectrogramVBO
    );


    // ========================================================
    // FFT
    // ========================================================

    std::array<float, FFT_SIZE> samples{};
    std::array<float, FFT_SIZE> windowFunction{};

    std::array<
        std::complex<float>,
        FFT_SIZE
    > fftValues{};

    for (int i = 0; i < FFT_SIZE; ++i)
    {
        windowFunction[i] =
            0.5f *
            (
                1.0f -
                std::cos(
                    2.0f * PI *
                    static_cast<float>(i) /
                    static_cast<float>(FFT_SIZE - 1)
                )
            );
    }


    // ========================================================
    // ESTADO
    // ========================================================

    float visualGain = 1.0f;
    int spectrogramColumn = 0;


    // ========================================================
    // ESTILO OPENGL
    // ========================================================

    glEnable(GL_BLEND);

    glBlendFunc(
        GL_SRC_ALPHA,
        GL_ONE_MINUS_SRC_ALPHA
    );

    glLineWidth(1.0f);


    // ========================================================
    // LOOP PRINCIPAL
    // ========================================================

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(
                window,
                GLFW_TRUE
            );
        }


        // ====================================================
        // VIEWPORT Y DISTRIBUCIÓN
        // ====================================================

        glfwGetFramebufferSize(
            window,
            &framebufferWidth,
            &framebufferHeight
        );

        glViewport(
            0,
            0,
            framebufferWidth,
            framebufferHeight
        );

        const float screenWidth =
            static_cast<float>(framebufferWidth);

        const float screenHeight =
            static_cast<float>(framebufferHeight);

        Rectangle waveformPanel{};
        Rectangle waveformPlot{};
        Rectangle spectrogramPanel{};
        Rectangle spectrogramPlot{};

        calculateLayout(
            framebufferWidth,
            framebufferHeight,
            waveformPanel,
            waveformPlot,
            spectrogramPanel,
            spectrogramPlot
        );


        // ====================================================
        // OBTENER AUDIO
        // ====================================================

        const int currentWriteIndex =
            audioData.writeIndex.load(
                std::memory_order_relaxed
            );

        for (int i = 0; i < FFT_SIZE; ++i)
        {
            const int index =
                (
                    (
                        currentWriteIndex -
                        FFT_SIZE +
                        i
                    ) %
                    WAVEFORM_SAMPLES +
                    WAVEFORM_SAMPLES
                ) %
                WAVEFORM_SAMPLES;

            samples[i] =
                audioData.samples[index].load(
                    std::memory_order_relaxed
                );
        }


        // ====================================================
        // GANANCIA VISUAL
        // ====================================================

        float peak = 0.0f;

        for (const float sample : samples)
        {
            peak = std::max(
                peak,
                std::abs(sample)
            );
        }

        float targetGain = MIN_GAIN;

        if (peak > 1.0e-6f)
        {
            targetGain = TARGET_AMPLITUDE / peak;

            targetGain = std::clamp(
                targetGain,
                MIN_GAIN,
                MAX_GAIN
            );
        }

        if (targetGain > visualGain)
        {
            visualGain +=
                (targetGain - visualGain) *
                GAIN_ATTACK;
        }
        else
        {
            visualGain +=
                (targetGain - visualGain) *
                GAIN_RELEASE;
        }


        // ====================================================
        // FFT
        // ====================================================

        for (int i = 0; i < FFT_SIZE; ++i)
        {
            fftValues[i] = std::complex<float>(
                samples[i] * windowFunction[i],
                0.0f
            );
        }

        fft(fftValues);


        // ====================================================
        // ESPECTROGRAMA
        // ====================================================

        for (int bin = 0; bin < SPECTRUM_BINS; ++bin)
        {
            const float magnitude =
                std::abs(fftValues[bin]);

            const float normalizedMagnitude =
                magnitude / static_cast<float>(FFT_SIZE);

            const float enhancedMagnitude =
                normalizedMagnitude *
                SPECTROGRAM_VISUAL_GAIN;

            const float safeMagnitude =
                std::max(
                    enhancedMagnitude,
                    1.0e-7f
                );

            float decibels =
                20.0f * std::log10(safeMagnitude);

            decibels = std::clamp(
                decibels,
                SPECTROGRAM_MIN_DB,
                SPECTROGRAM_MAX_DB
            );

            float normalized =
                (
                    decibels -
                    SPECTROGRAM_MIN_DB
                ) /
                (
                    SPECTROGRAM_MAX_DB -
                    SPECTROGRAM_MIN_DB
                );

            normalized = std::clamp(
                normalized,
                0.0f,
                1.0f
            );

            spectrogramData[
                bin * SPECTROGRAM_WIDTH +
                spectrogramColumn
            ] =
                static_cast<unsigned char>(
                    normalized * 255.0f
                );
        }

        spectrogramColumn =
            (spectrogramColumn + 1) %
            SPECTROGRAM_WIDTH;


        // ====================================================
        // ACTUALIZAR TEXTURA
        // ====================================================

        glBindTexture(
            GL_TEXTURE_2D,
            spectrogramTexture
        );

        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            SPECTROGRAM_WIDTH,
            SPECTROGRAM_HEIGHT,
            GL_RED,
            GL_UNSIGNED_BYTE,
            spectrogramData.data()
        );


        // ====================================================
        // ACTUALIZAR QUAD
        // ====================================================

        updateSpectrogramQuad(
            spectrogramVBO,
            spectrogramPlot,
            screenWidth,
            screenHeight
        );


        // ====================================================
        // LIMPIAR PANTALLA
        // ====================================================

        glClearColor(
            0.015f,
            0.020f,
            0.030f,
            1.0f
        );

        glClear(GL_COLOR_BUFFER_BIT);


        // ====================================================
        // DIBUJAR ESPECTROGRAMA
        // ====================================================

        drawSpectrogram(
            spectrumShaderProgram,
            spectrogramVAO,
            spectrogramTexture,
            spectrumLocation,
            columnLocation,
            spectrogramColumn
        );


        // ====================================================
        // DIBUJAR ONDA
        // ====================================================

        glUseProgram(lineShaderProgram);

        drawWaveform(
            waveformVAO,
            waveformVBO,
            samples,
            visualGain,
            waveformPlot,
            screenWidth,
            screenHeight
        );


        // ====================================================
        // MARCOS, EJES Y CUADRÍCULAS
        // ====================================================

        drawInterface(
            lineRenderer,
            waveformPanel,
            waveformPlot,
            spectrogramPanel,
            spectrogramPlot,
            screenWidth,
            screenHeight
        );


        // ====================================================
        // ETIQUETAS
        // ====================================================

        drawWaveformLabels(
            waveformPanel,
            waveformPlot,
            screenWidth,
            screenHeight
        );

        drawSpectrogramLabels(
            spectrogramPanel,
            spectrogramPlot,
            screenWidth,
            screenHeight
        );


        // ====================================================
        // PRESENTAR
        // ====================================================

        glfwSwapBuffers(window);
    }


    // ========================================================
    // LIMPIEZA
    // ========================================================

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

    glDeleteVertexArrays(1, &textVAO);
    glDeleteBuffers(1, &textVBO);

    glDeleteVertexArrays(1, &waveformVAO);
    glDeleteBuffers(1, &waveformVBO);

    glDeleteVertexArrays(1, &spectrogramVAO);
    glDeleteBuffers(1, &spectrogramVBO);

    glDeleteVertexArrays(1, &lineRenderer.vao);
    glDeleteBuffers(1, &lineRenderer.vbo);

    glDeleteTextures(1, &spectrogramTexture);

    glDeleteProgram(lineShaderProgram);
    glDeleteProgram(spectrumShaderProgram);
    glDeleteProgram(textShaderProgram);

    glfwDestroyWindow(window);
    glfwTerminate();

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    return 0;
}