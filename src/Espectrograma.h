#pragma once

#include <array>
#include <complex>

#include <glad/glad.h>

#include "AudioController.h"
#include "Renderer.h"

// ============================================================
// CONFIGURACIÓN DEL ESPECTROGRAMA
// ============================================================

constexpr int FFT_SIZE = 1024;

constexpr int SPECTRUM_BINS =
    FFT_SIZE / 2;

constexpr int SPECTROGRAM_WIDTH = 1024;

constexpr int SPECTROGRAM_HEIGHT =
    SPECTRUM_BINS;

constexpr float SPECTROGRAM_MIN_DB = -100.0f;

constexpr float SPECTROGRAM_MAX_DB = 0.0f;

constexpr float SPECTROGRAM_VISUAL_GAIN = 4.0f;

constexpr float PI =
    3.14159265358979323846f;

constexpr float SPECTROGRAM_DURATION =
    static_cast<float>(SPECTROGRAM_WIDTH) *
    static_cast<float>(FRAMES_PER_BUFFER) /
    static_cast<float>(SAMPLE_RATE);

// ============================================================
// FFT
// ============================================================

void fft(
    std::array<
        std::complex<float>,
        FFT_SIZE
    >& values
);

// ============================================================
// TEXTURA DEL ESPECTROGRAMA
// ============================================================

GLuint createSpectrogramTexture(
    const std::array<
        unsigned char,
        SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT
    >& data
);

// ============================================================
// DIBUJAR ESPECTROGRAMA
// ============================================================

void drawSpectrogram(
    GLuint shader,
    GLuint vao,
    GLuint texture,
    GLint spectrumLocation,
    GLint columnLocation,
    int column
);