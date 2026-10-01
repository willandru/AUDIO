#pragma once

#include <array>

#include <glad/glad.h>

#include "AudioController.h"
#include "Renderer.h"


constexpr float WAVEFORM_DURATION =
    static_cast<float>(WAVEFORM_SAMPLES) /
    static_cast<float>(SAMPLE_RATE);


void drawWaveform(
    GLuint vao,
    GLuint vbo,
    const std::array<float, WAVEFORM_SAMPLES>& samples,
    float gain,
    const Rectangle& plot,
    float screenWidth,
    float screenHeight
);