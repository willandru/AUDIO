#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>

#include "AudioController.h"
#include "Camera.h"
#include "Espectrograma.h"
#include "InputKeyboard.h"
#include "InputMouse.h"
#include "Renderer.h"
#include "Shader.h"
#include "TextRenderer.h"
#include "WaveForm.h"
#include "Window.h"

// ============================================================
// CONFIGURACIÓN
// ============================================================

constexpr float TARGET_AMPLITUDE = 0.65f;

constexpr float GAIN_ATTACK = 0.20f;

constexpr float GAIN_RELEASE = 0.02f;

constexpr float MIN_GAIN = 1.0f;

constexpr float MAX_GAIN = 50.0f;


// ============================================================
// MAIN
// ============================================================

int main()
{
    // ========================================================
    // AUDIO
    // ========================================================

    AudioController audioController;

    if (!audioController.initialize())
    {
        return 1;
    }

    if (!audioController.start())
    {
        return 1;
    }


    // ========================================================
    // VENTANA
    // ========================================================

    Window window;

    if (!window.initialize())
    {
        audioController.stop();

        return 1;
    }


    // ========================================================
    // ENTRADA
    // ========================================================

    InputKeyboard keyboard;

    InputMouse mouse;

    Camera camera;


    // ========================================================
    // FUENTE
    // ========================================================

    TextRenderer textRenderer;

    if (!textRenderer.initialize())
    {
        window.close();
        audioController.stop();

        return 1;
    }


    // ========================================================
    // SHADER DEL ESPECTROGRAMA
    // ========================================================

    const GLuint spectrumShaderProgram =
        Shader::createProgramFromFiles(
            "../src/shaders/spectrum.vert",
            "../src/shaders/spectrum.frag"
        );

    if (spectrumShaderProgram == 0)
    {
        textRenderer.cleanup();
        window.close();
        audioController.stop();

        return 1;
    }

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

    glGenVertexArrays(
        1,
        &waveformVAO
    );

    glGenBuffers(
        1,
        &waveformVBO
    );

    glBindVertexArray(
        waveformVAO
    );

    glBindBuffer(
        GL_ARRAY_BUFFER,
        waveformVBO
    );

    glBufferData(
        GL_ARRAY_BUFFER,
        WAVEFORM_SAMPLES *
            2 *
            sizeof(float),
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
    // RENDERER DE LÍNEAS
    // ========================================================

    LineRenderer lineRenderer;

    initializeLineRenderer(
        lineRenderer
    );


    // ========================================================
    // TEXTURA DEL ESPECTROGRAMA
    // ========================================================

    std::array<
        unsigned char,
        SPECTROGRAM_WIDTH *
        SPECTROGRAM_HEIGHT
    > spectrogramData{};

    const GLuint spectrogramTexture =
        createSpectrogramTexture(
            spectrogramData
        );


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

    std::array<
        float,
        FFT_SIZE
    > samples{};

    std::array<
        float,
        FFT_SIZE
    > windowFunction{};

    std::array<
        std::complex<float>,
        FFT_SIZE
    > fftValues{};

    for (int i = 0;
         i < FFT_SIZE;
         ++i)
    {
        windowFunction[i] =
            0.5f *
            (
                1.0f -
                std::cos(
                    2.0f *
                    PI *
                    static_cast<float>(i) /
                    static_cast<float>(
                        FFT_SIZE - 1
                    )
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

    while (!window.shouldClose())
    {
        window.pollEvents();

        keyboard.process(
            window.getHandle()
        );

        mouse.process(
            window.getHandle()
        );

        (void)camera;


        // ====================================================
        // VIEWPORT
        // ====================================================

        window.updateViewport();

        const float screenWidth =
            static_cast<float>(
                window.getWidth()
            );

        const float screenHeight =
            static_cast<float>(
                window.getHeight()
            );


        // ====================================================
        // DISTRIBUCIÓN DE PANELES
        // ====================================================

        Rectangle waveformPanel{};

        Rectangle waveformPlot{};

        Rectangle spectrogramPanel{};

        Rectangle spectrogramPlot{};

        const float margin =
            std::max(
                18.0f,
                screenWidth * 0.018f
            );

        const float gap =
            std::max(
                12.0f,
                screenHeight * 0.018f
            );

        const float availableHeight =
            screenHeight -
            2.0f * margin -
            gap;

        const float waveformHeight =
            availableHeight * 0.43f;

        const float spectrogramHeight =
            availableHeight -
            waveformHeight;

        waveformPanel =
        {
            margin,
            margin,
            screenWidth -
                2.0f * margin,
            waveformHeight
        };

        spectrogramPanel =
        {
            margin,
            margin +
                waveformHeight +
                gap,
            screenWidth -
                2.0f * margin,
            spectrogramHeight
        };

        const float leftAxis = 72.0f;

        const float rightMargin = 24.0f;

        const float topMargin = 46.0f;

        const float bottomMargin = 52.0f;

        waveformPlot =
        {
            waveformPanel.x +
                leftAxis,

            waveformPanel.y +
                topMargin,

            waveformPanel.width -
                leftAxis -
                rightMargin,

            waveformPanel.height -
                topMargin -
                bottomMargin
        };

        spectrogramPlot =
        {
            spectrogramPanel.x +
                leftAxis,

            spectrogramPanel.y +
                topMargin,

            spectrogramPanel.width -
                leftAxis -
                rightMargin -
                54.0f,

            spectrogramPanel.height -
                topMargin -
                bottomMargin
        };


        // ====================================================
        // OBTENER AUDIO
        // ====================================================

        const AudioData& audioData =
            audioController.getAudioData();

        const int currentWriteIndex =
            audioData.writeIndex.load(
                std::memory_order_relaxed
            );

        for (int i = 0;
             i < FFT_SIZE;
             ++i)
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
            peak =
                std::max(
                    peak,
                    std::abs(sample)
                );
        }

        float targetGain = MIN_GAIN;

        if (peak > 1.0e-6f)
        {
            targetGain =
                TARGET_AMPLITUDE /
                peak;

            targetGain =
                std::clamp(
                    targetGain,
                    MIN_GAIN,
                    MAX_GAIN
                );
        }

        if (targetGain > visualGain)
        {
            visualGain +=
                (
                    targetGain -
                    visualGain
                ) *
                GAIN_ATTACK;
        }
        else
        {
            visualGain +=
                (
                    targetGain -
                    visualGain
                ) *
                GAIN_RELEASE;
        }


        // ====================================================
        // FFT
        // ====================================================

        for (int i = 0;
             i < FFT_SIZE;
             ++i)
        {
            fftValues[i] =
                std::complex<float>(
                    samples[i] *
                        windowFunction[i],
                    0.0f
                );
        }

        fft(
            fftValues
        );


        // ====================================================
        // ESPECTROGRAMA
        // ====================================================

        for (int bin = 0;
             bin < SPECTRUM_BINS;
             ++bin)
        {
            const float magnitude =
                std::abs(
                    fftValues[bin]
                );

            const float normalizedMagnitude =
                magnitude /
                static_cast<float>(
                    FFT_SIZE
                );

            const float enhancedMagnitude =
                normalizedMagnitude *
                SPECTROGRAM_VISUAL_GAIN;

            const float safeMagnitude =
                std::max(
                    enhancedMagnitude,
                    1.0e-7f
                );

            float decibels =
                20.0f *
                std::log10(
                    safeMagnitude
                );

            decibels =
                std::clamp(
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

            normalized =
                std::clamp(
                    normalized,
                    0.0f,
                    1.0f
                );

            spectrogramData[
                bin *
                    SPECTROGRAM_WIDTH +
                spectrogramColumn
            ] =
                static_cast<unsigned char>(
                    normalized *
                    255.0f
                );
        }

        spectrogramColumn =
            (
                spectrogramColumn +
                1
            ) %
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

        glClear(
            GL_COLOR_BUFFER_BIT
        );


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
        // DIBUJAR FORMA DE ONDA
        // ====================================================

        glUseProgram(
            lineRenderer.shader
        );

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
        // INTERFAZ
        // ====================================================

        lineRenderer.vertices.clear();

        addRectangle(
            lineRenderer,
            waveformPanel,
            screenWidth,
            screenHeight
        );

        addRectangle(
            lineRenderer,
            spectrogramPanel,
            screenWidth,
            screenHeight
        );

        addRectangle(
            lineRenderer,
            waveformPlot,
            screenWidth,
            screenHeight
        );

        addRectangle(
            lineRenderer,
            spectrogramPlot,
            screenWidth,
            screenHeight
        );

        addPlotGrid(
            lineRenderer,
            waveformPlot,
            4,
            10,
            screenWidth,
            screenHeight
        );

        addPlotGrid(
            lineRenderer,
            spectrogramPlot,
            5,
            10,
            screenWidth,
            screenHeight
        );

        addAxisTicks(
            lineRenderer,
            waveformPlot,
            4,
            10,
            screenWidth,
            screenHeight
        );

        addAxisTicks(
            lineRenderer,
            spectrogramPlot,
            5,
            10,
            screenWidth,
            screenHeight
        );

        drawLines(
            lineRenderer
        );


        // ====================================================
        // ETIQUETAS — FORMA DE ONDA
        // ====================================================

        const float textScale =
            std::clamp(
                screenHeight / 900.0f,
                0.8f,
                1.5f
            );

        textRenderer.render(
            "FORMA DE ONDA",
            waveformPanel.x + 30.0f,
            waveformPanel.y + 10.0f,
            textScale * 1.35f,
            screenWidth,
            screenHeight,
            0.92f,
            0.95f,
            0.98f
        );

        textRenderer.render(
            "Amplitud",
            waveformPanel.x + 14.0f,
            waveformPlot.y +
                waveformPlot.height * 0.5f -
                10.0f,
            textScale,
            screenWidth,
            screenHeight
        );

        textRenderer.render(
            "Tiempo (ms)",
            waveformPlot.x +
                waveformPlot.width * 0.5f -
                40.0f,
            waveformPlot.y +
                waveformPlot.height +
                28.0f,
            textScale,
            screenWidth,
            screenHeight
        );

        constexpr int WAVEFORM_Y_DIVISIONS = 4;

        for (int i = 0;
             i <= WAVEFORM_Y_DIVISIONS;
             ++i)
        {
            const float value =
                -1.0f +
                2.0f *
                static_cast<float>(i) /
                static_cast<float>(
                    WAVEFORM_Y_DIVISIONS
                );

            const float y =
                waveformPlot.y +
                waveformPlot.height -
                waveformPlot.height *
                static_cast<float>(i) /
                static_cast<float>(
                    WAVEFORM_Y_DIVISIONS
                );

            char label[32];

            std::snprintf(
                label,
                sizeof(label),
                "%.1f",
                value
            );

            textRenderer.render(
                label,
                waveformPlot.x - 38.0f,
                y - 9.0f,
                textScale * 0.9f,
                screenWidth,
                screenHeight
            );
        }

        constexpr int WAVEFORM_X_DIVISIONS = 10;

        for (int i = 0;
             i <= WAVEFORM_X_DIVISIONS;
             ++i)
        {
            const float fraction =
                static_cast<float>(i) /
                static_cast<float>(
                    WAVEFORM_X_DIVISIONS
                );

            const float x =
                waveformPlot.x +
                waveformPlot.width *
                fraction;

            const float milliseconds =
                fraction *
                WAVEFORM_DURATION *
                1000.0f;

            char label[32];

            std::snprintf(
                label,
                sizeof(label),
                "%.1f",
                milliseconds
            );

            textRenderer.render(
                label,
                x - 12.0f,
                waveformPlot.y +
                    waveformPlot.height +
                    5.0f,
                textScale * 0.85f,
                screenWidth,
                screenHeight
            );
        }


        // ====================================================
        // ETIQUETAS — ESPECTROGRAMA
        // ====================================================

        textRenderer.render(
            "ESPECTROGRAMA",
            spectrogramPanel.x +
                spectrogramPanel.width * 0.5f -
                65.0f,
            spectrogramPanel.y + 10.0f,
            textScale * 1.35f,
            screenWidth,
            screenHeight,
            0.92f,
            0.95f,
            0.98f
        );

        textRenderer.render(
            "Frecuencia (Hz)",
            spectrogramPanel.x + 12.0f,
            spectrogramPanel.y + 2.0f,
            textScale,
            screenWidth,
            screenHeight
        );

        textRenderer.render(
            "Tiempo (s)",
            spectrogramPlot.x +
                spectrogramPlot.width * 0.5f -
                30.0f,
            spectrogramPlot.y +
                spectrogramPlot.height +
                28.0f,
            textScale,
            screenWidth,
            screenHeight
        );

        constexpr int SPECTROGRAM_Y_DIVISIONS = 5;

        for (int i = 0;
             i <= SPECTROGRAM_Y_DIVISIONS;
             ++i)
        {
            const float fraction =
                static_cast<float>(i) /
                static_cast<float>(
                    SPECTROGRAM_Y_DIVISIONS
                );

            const float frequency =
                fraction *
                SAMPLE_RATE *
                0.5f;

            const float y =
                spectrogramPlot.y +
                spectrogramPlot.height -
                spectrogramPlot.height *
                fraction;

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

            textRenderer.render(
                label,
                spectrogramPlot.x - 47.0f,
                y - 9.0f,
                textScale * 0.9f,
                screenWidth,
                screenHeight
            );
        }

        constexpr int SPECTROGRAM_X_DIVISIONS = 10;

        for (int i = 0;
             i <= SPECTROGRAM_X_DIVISIONS;
             ++i)
        {
            const float fraction =
                static_cast<float>(i) /
                static_cast<float>(
                    SPECTROGRAM_X_DIVISIONS
                );

            const float x =
                spectrogramPlot.x +
                spectrogramPlot.width *
                fraction;

            const float seconds =
                fraction *
                SPECTROGRAM_DURATION;

            char label[32];

            std::snprintf(
                label,
                sizeof(label),
                "%.1f",
                seconds
            );

            textRenderer.render(
                label,
                x - 10.0f,
                spectrogramPlot.y +
                    spectrogramPlot.height +
                    5.0f,
                textScale * 0.85f,
                screenWidth,
                screenHeight
            );
        }


        // ====================================================
        // PRESENTAR
        // ====================================================

        window.swapBuffers();
    }


    // ========================================================
    // LIMPIEZA
    // ========================================================

    textRenderer.cleanup();

    glDeleteVertexArrays(
        1,
        &waveformVAO
    );

    glDeleteBuffers(
        1,
        &waveformVBO
    );

    glDeleteVertexArrays(
        1,
        &spectrogramVAO
    );

    glDeleteBuffers(
        1,
        &spectrogramVBO
    );

    glDeleteVertexArrays(
        1,
        &lineRenderer.vao
    );

    glDeleteBuffers(
        1,
        &lineRenderer.vbo
    );

    glDeleteTextures(
        1,
        &spectrogramTexture
    );

    glDeleteProgram(
        lineRenderer.shader
    );

    glDeleteProgram(
        spectrumShaderProgram
    );

    window.close();

    audioController.stop();

    return 0;
}