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
#include "Spectrogram3DRenderer.h"
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
// DIAGNÓSTICO DE FRECUENCIAS
// ============================================================

constexpr float DIAGNOSTIC_INTERVAL = 1.0f;

constexpr int DIAGNOSTIC_BANDS = 6;

constexpr float DIAGNOSTIC_BAND_LIMITS[
    DIAGNOSTIC_BANDS + 1
] =
{
    0.0f,
    1000.0f,
    5000.0f,
    10000.0f,
    15000.0f,
    20000.0f,
    22050.0f
};


// ============================================================
// VISUALIZACIÓN 2D SUPERIOR
// ============================================================

enum class TopVisualization
{
    Waveform,
    Spectrogram
};


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
    // SHADER DEL ESPECTROGRAMA 2D
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
    // SHADER DEL ESPECTROGRAMA 3D
    // ========================================================

    const GLuint spectrogram3DShaderProgram =
        Shader::createProgramFromFiles(
            "../src/shaders/spectrogram3d.vert",
            "../src/shaders/spectrogram3d.frag"
        );

    if (spectrogram3DShaderProgram == 0)
    {
        glDeleteProgram(
            spectrumShaderProgram
        );

        textRenderer.cleanup();

        window.close();

        audioController.stop();

        return 1;
    }


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
    // RENDERER DEL ESPECTROGRAMA 3D
    // ========================================================

    Spectrogram3DRenderer spectrogram3DRenderer;

    if (
        !spectrogram3DRenderer.initialize(
            SPECTROGRAM_WIDTH,
            SPECTROGRAM_HEIGHT
        )
    )
    {
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

        glDeleteProgram(
            spectrogram3DShaderProgram
        );

        window.close();

        audioController.stop();

        return 1;
    }


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

    TopVisualization topVisualization =
        TopVisualization::Waveform;


    // ========================================================
    // ESTADO DEL DIAGNÓSTICO
    // ========================================================

    float diagnosticTime = 0.0f;

    std::array<
        double,
        DIAGNOSTIC_BANDS
    > diagnosticEnergy{};


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


        // ====================================================
        // TECLADO
        // ====================================================

        keyboard.process(
            window.getHandle()
        );


        // ====================================================
        // CAMBIO DE VISUALIZACIÓN 2D
        // ====================================================

        if (keyboard.isLeftPressed())
        {
            if (
                topVisualization ==
                TopVisualization::Waveform
            )
            {
                topVisualization =
                    TopVisualization::Spectrogram;
            }
            else
            {
                topVisualization =
                    TopVisualization::Waveform;
            }
        }


        if (keyboard.isRightPressed())
        {
            if (
                topVisualization ==
                TopVisualization::Waveform
            )
            {
                topVisualization =
                    TopVisualization::Spectrogram;
            }
            else
            {
                topVisualization =
                    TopVisualization::Waveform;
            }
        }


        // ====================================================
        // MOUSE
        // ====================================================

        mouse.process(
            window.getHandle()
        );

        (void)camera;

        (void)keyboard.isAPressed();
        (void)keyboard.isDPressed();


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


        const float topPanelHeight =
            availableHeight * 0.43f;


        const float bottomPanelHeight =
            availableHeight -
            topPanelHeight;


        Rectangle topPanel =
        {
            margin,
            margin,
            screenWidth -
                2.0f * margin,
            topPanelHeight
        };


        Rectangle bottomPanel =
        {
            margin,
            margin +
                topPanelHeight +
                gap,
            screenWidth -
                2.0f * margin,
            bottomPanelHeight
        };


        const float leftAxis = 72.0f;

        const float rightMargin = 24.0f;

        const float topMargin = 46.0f;

        const float bottomMargin = 52.0f;


        Rectangle topPlot =
        {
            topPanel.x +
                leftAxis,

            topPanel.y +
                topMargin,

            topPanel.width -
                leftAxis -
                rightMargin,

            topPanel.height -
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
        // DIAGNÓSTICO DE BANDAS
        // ====================================================

        for (int bin = 0;
             bin < SPECTRUM_BINS;
             ++bin)
        {
            const float frequency =
                static_cast<float>(bin) *
                static_cast<float>(SAMPLE_RATE) /
                static_cast<float>(FFT_SIZE);


            const float magnitude =
                std::abs(
                    fftValues[bin]
                );


            const double power =
                static_cast<double>(
                    magnitude
                ) *
                static_cast<double>(
                    magnitude
                );


            for (int band = 0;
                 band < DIAGNOSTIC_BANDS;
                 ++band)
            {
                if (
                    frequency >=
                        DIAGNOSTIC_BAND_LIMITS[band] &&
                    frequency <
                        DIAGNOSTIC_BAND_LIMITS[band + 1]
                )
                {
                    diagnosticEnergy[band] +=
                        power;

                    break;
                }
            }
        }


        diagnosticTime +=
            static_cast<float>(
                FRAMES_PER_BUFFER
            ) /
            static_cast<float>(
                SAMPLE_RATE
            );


        if (
            diagnosticTime >=
            DIAGNOSTIC_INTERVAL
        )
        {
            std::cout
                << "\n========================================\n"
                << "ENERGÍA DEL AUDIO POR BANDA\n"
                << "========================================\n";


            constexpr const char* bandNames[
                DIAGNOSTIC_BANDS
            ] =
            {
                "0 - 1 kHz",
                "1 - 5 kHz",
                "5 - 10 kHz",
                "10 - 15 kHz",
                "15 - 20 kHz",
                "20 - 22.05 kHz"
            };


            double totalEnergy = 0.0;


            for (int band = 0;
                 band < DIAGNOSTIC_BANDS;
                 ++band)
            {
                totalEnergy +=
                    diagnosticEnergy[band];
            }


            for (int band = 0;
                 band < DIAGNOSTIC_BANDS;
                 ++band)
            {
                double percentage = 0.0;


                if (totalEnergy > 0.0)
                {
                    percentage =
                        100.0 *
                        diagnosticEnergy[band] /
                        totalEnergy;
                }


                std::cout
                    << bandNames[band]
                    << ": "
                    << percentage
                    << " %\n";
            }


            std::cout
                << "========================================\n";


            diagnosticEnergy.fill(
                0.0
            );


            diagnosticTime = 0.0f;
        }


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


        // ====================================================
        // COLUMNA ACTUAL DEL ESPECTROGRAMA
        // ====================================================

        const int currentSpectrogramColumn =
            spectrogramColumn;


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
        // ACTUALIZAR 3D
        // ====================================================

        spectrogram3DRenderer.update(
            spectrogramData.data(),
            SPECTROGRAM_WIDTH,
            SPECTROGRAM_HEIGHT,
            currentSpectrogramColumn
        );


        // ====================================================
        // ACTUALIZAR QUAD 2D
        // ====================================================

        updateSpectrogramQuad(
            spectrogramVBO,
            topPlot,
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
            GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT
        );


        // ====================================================
        // PANEL SUPERIOR — 2D
        // ====================================================

        if (
            topVisualization ==
            TopVisualization::Spectrogram
        )
        {
            drawSpectrogram(
                spectrumShaderProgram,
                spectrogramVAO,
                spectrogramTexture,
                spectrumLocation,
                columnLocation,
                spectrogramColumn
            );
        }
        else
        {
            glUseProgram(
                lineRenderer.shader
            );


            drawWaveform(
                waveformVAO,
                waveformVBO,
                samples,
                visualGain,
                topPlot,
                screenWidth,
                screenHeight
            );
        }


        // ====================================================
        // PANEL INFERIOR — 3D
        // ====================================================

        glEnable(
            GL_DEPTH_TEST
        );


        glViewport(
            static_cast<GLint>(
                bottomPanel.x
            ),
            static_cast<GLint>(
                screenHeight -
                bottomPanel.y -
                bottomPanel.height
            ),
            static_cast<GLsizei>(
                bottomPanel.width
            ),
            static_cast<GLsizei>(
                bottomPanel.height
            )
        );


        spectrogram3DRenderer.render(
            spectrogram3DShaderProgram
        );


        glDisable(
            GL_DEPTH_TEST
        );


        window.updateViewport();


        // ====================================================
        // INTERFAZ 2D
        // ====================================================

        lineRenderer.vertices.clear();


        addRectangle(
            lineRenderer,
            topPanel,
            screenWidth,
            screenHeight
        );


        addRectangle(
            lineRenderer,
            bottomPanel,
            screenWidth,
            screenHeight
        );


        addRectangle(
            lineRenderer,
            topPlot,
            screenWidth,
            screenHeight
        );


        addPlotGrid(
            lineRenderer,
            topPlot,
            4,
            10,
            screenWidth,
            screenHeight
        );


        addAxisTicks(
            lineRenderer,
            topPlot,
            4,
            10,
            screenWidth,
            screenHeight
        );


        drawLines(
            lineRenderer
        );


        // ====================================================
        // TEXTO
        // ====================================================

        const float textScale =
            std::clamp(
                screenHeight / 900.0f,
                0.8f,
                1.5f
            );


        // ====================================================
        // WAVEFORM
        // ====================================================

        if (
            topVisualization ==
            TopVisualization::Waveform
        )
        {
            textRenderer.render(
                "FORMA DE ONDA",
                topPanel.x + 30.0f,
                topPanel.y + 10.0f,
                textScale * 1.35f,
                screenWidth,
                screenHeight,
                0.92f,
                0.95f,
                0.98f
            );


            textRenderer.render(
                "Amplitud",
                topPanel.x + 14.0f,
                topPlot.y +
                    topPlot.height * 0.5f -
                    10.0f,
                textScale,
                screenWidth,
                screenHeight
            );


            textRenderer.render(
                "Tiempo (ms)",
                topPlot.x +
                    topPlot.width * 0.5f -
                    40.0f,
                topPlot.y +
                    topPlot.height +
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
                    topPlot.y +
                    topPlot.height -
                    topPlot.height *
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
                    topPlot.x - 38.0f,
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
                    topPlot.x +
                    topPlot.width *
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
                    topPlot.y +
                        topPlot.height +
                        5.0f,
                    textScale * 0.85f,
                    screenWidth,
                    screenHeight
                );
            }
        }


        // ====================================================
        // ESPECTROGRAMA 2D
        // ====================================================

        else
        {
            textRenderer.render(
                "ESPECTROGRAMA",
                topPanel.x +
                    topPanel.width * 0.5f -
                    65.0f,
                topPanel.y + 10.0f,
                textScale * 1.35f,
                screenWidth,
                screenHeight,
                0.92f,
                0.95f,
                0.98f
            );


            textRenderer.render(
                "Frecuencia (Hz)",
                topPanel.x + 12.0f,
                topPanel.y + 2.0f,
                textScale,
                screenWidth,
                screenHeight
            );


            textRenderer.render(
                "Tiempo (s)",
                topPlot.x +
                    topPlot.width * 0.5f -
                    30.0f,
                topPlot.y +
                    topPlot.height +
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
                    topPlot.y +
                    topPlot.height -
                    topPlot.height *
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
                    topPlot.x - 47.0f,
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
                    topPlot.x +
                    topPlot.width *
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
                    topPlot.y +
                        topPlot.height +
                        5.0f,
                    textScale * 0.85f,
                    screenWidth,
                    screenHeight
                );
            }
        }


        // ====================================================
        // TÍTULO 3D
        // ====================================================

        textRenderer.render(
            "ESPECTROGRAMA 3D",
            bottomPanel.x +
                bottomPanel.width * 0.5f -
                80.0f,
            bottomPanel.y + 10.0f,
            textScale * 1.35f,
            screenWidth,
            screenHeight,
            0.92f,
            0.95f,
            0.98f
        );


        // ====================================================
        // PRESENTAR
        // ====================================================

        window.swapBuffers();
    }


    // ========================================================
    // LIMPIEZA
    // ========================================================

    spectrogram3DRenderer.destroy();


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


    glDeleteProgram(
        spectrogram3DShaderProgram
    );


    window.close();


    audioController.stop();


    return 0;
}