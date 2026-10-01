#include "Espectrograma.h"

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

    for (int i = 1;
         i < FFT_SIZE;
         ++i)
    {
        int bit =
            FFT_SIZE >> 1;

        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j)
        {
            std::swap(
                values[i],
                values[j]
            );
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

        for (int i = 0;
             i < FFT_SIZE;
             i += length)
        {
            std::complex<float> w(
                1.0f,
                0.0f
            );

            const int halfLength =
                length / 2;

            for (int j = 0;
                 j < halfLength;
                 ++j)
            {
                const auto u =
                    values[i + j];

                const auto v =
                    values[
                        i + j + halfLength
                    ] * w;

                values[i + j] =
                    u + v;

                values[
                    i + j + halfLength
                ] =
                    u - v;

                w *= wlen;
            }
        }
    }
}

// ============================================================
// CREAR TEXTURA DEL ESPECTROGRAMA
// ============================================================

GLuint createSpectrogramTexture(
    const std::array<
        unsigned char,
        SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT
    >& data)
{
    GLuint texture = 0;

    glGenTextures(
        1,
        &texture
    );

    glBindTexture(
        GL_TEXTURE_2D,
        texture
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

    glActiveTexture(
        GL_TEXTURE0
    );

    glBindTexture(
        GL_TEXTURE_2D,
        texture
    );

    glUniform1i(
        spectrumLocation,
        0
    );

    glUniform1f(
        columnLocation,
        static_cast<float>(column)
    );

    glBindVertexArray(
        vao
    );

    glDrawArrays(
        GL_TRIANGLES,
        0,
        6
    );

    glBindVertexArray(0);
}