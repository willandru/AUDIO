#include "WaveForm.h"

#include <algorithm>


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
        const float fraction =
            static_cast<float>(i) /
            static_cast<float>(WAVEFORM_SAMPLES - 1);

        const float x =
            plot.x +
            fraction * plot.width;

        const float value =
            std::clamp(
                samples[i] * gain,
                -1.0f,
                1.0f
            );

        const float y =
            plot.y +
            plot.height * 0.5f -
            value * plot.height * 0.5f;

        const Point point =
            screenToNDC(
                x,
                y,
                screenWidth,
                screenHeight
            );

        vertices[i * 2] = point.x;
        vertices[i * 2 + 1] = point.y;
    }

    glBindBuffer(
        GL_ARRAY_BUFFER,
        vbo
    );

    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        sizeof(vertices),
        vertices.data()
    );

    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );

    glBindVertexArray(vao);

    glDrawArrays(
        GL_LINE_STRIP,
        0,
        WAVEFORM_SAMPLES
    );

    glBindVertexArray(0);
}