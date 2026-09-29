#include <algorithm>
#include <array>
#include <atomic>
#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <portaudio.h>

constexpr int SAMPLE_RATE = 44100;
constexpr int FRAMES_PER_BUFFER = 256;
constexpr int CHANNELS = 1;

constexpr int WAVEFORM_SAMPLES = 1024;

constexpr float WAVEFORM_SCALE = 6.0f;

struct AudioData
{
    std::array<std::atomic<float>, WAVEFORM_SAMPLES> samples;
    std::atomic<unsigned int> writeIndex{0};

    AudioData()
    {
        for (auto& sample : samples)
        {
            sample.store(0.0f);
        }
    }
};

int audioCallback(
    const void* inputBuffer,
    void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData)
{
    (void)outputBuffer;
    (void)timeInfo;
    (void)statusFlags;

    auto* audioData =
        static_cast<AudioData*>(userData);

    const auto* input =
        static_cast<const float*>(inputBuffer);

    if (input == nullptr)
    {
        return paContinue;
    }

    unsigned int writeIndex =
        audioData->writeIndex.load(
            std::memory_order_relaxed
        );

    for (unsigned long i = 0;
         i < framesPerBuffer;
         ++i)
    {
        audioData->samples[writeIndex].store(
            input[i],
            std::memory_order_relaxed
        );

        writeIndex =
            (writeIndex + 1) % WAVEFORM_SAMPLES;
    }

    audioData->writeIndex.store(
        writeIndex,
        std::memory_order_relaxed
    );

    return paContinue;
}

const char* vertexShaderSource = R"(
#version 330 core

layout (location = 0) in vec2 aPosition;

void main()
{
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 330 core

out vec4 FragColor;

void main()
{
    FragColor = vec4(0.2, 0.8, 1.0, 1.0);
}
)";

GLuint compileShader(
    GLenum type,
    const char* source)
{
    const GLuint shader =
        glCreateShader(type);

    glShaderSource(
        shader,
        1,
        &source,
        nullptr
    );

    glCompileShader(shader);

    GLint success = 0;

    glGetShaderiv(
        shader,
        GL_COMPILE_STATUS,
        &success
    );

    if (!success)
    {
        char infoLog[512];

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

        glDeleteShader(shader);

        return 0;
    }

    return shader;
}

GLuint createShaderProgram()
{
    const GLuint vertexShader =
        compileShader(
            GL_VERTEX_SHADER,
            vertexShaderSource
        );

    if (vertexShader == 0)
    {
        return 0;
    }

    const GLuint fragmentShader =
        compileShader(
            GL_FRAGMENT_SHADER,
            fragmentShaderSource
        );

    if (fragmentShader == 0)
    {
        glDeleteShader(vertexShader);

        return 0;
    }

    const GLuint program =
        glCreateProgram();

    glAttachShader(
        program,
        vertexShader
    );

    glAttachShader(
        program,
        fragmentShader
    );

    glLinkProgram(program);

    GLint success = 0;

    glGetProgramiv(
        program,
        GL_LINK_STATUS,
        &success
    );

    if (!success)
    {
        char infoLog[512];

        glGetProgramInfoLog(
            program,
            sizeof(infoLog),
            nullptr,
            infoLog
        );

        std::cerr
            << "Error enlazando shader program:\n"
            << infoLog
            << '\n';

        glDeleteProgram(program);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return 0;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

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

    const PaDeviceIndex deviceIndex =
        Pa_GetDefaultInputDevice();

    if (deviceIndex == paNoDevice)
    {
        std::cerr
            << "No se encontro un dispositivo de entrada.\n";

        Pa_Terminate();

        return 1;
    }

    const PaDeviceInfo* deviceInfo =
        Pa_GetDeviceInfo(deviceIndex);

    if (deviceInfo == nullptr)
    {
        std::cerr
            << "No se pudo obtener informacion del dispositivo.\n";

        Pa_Terminate();

        return 1;
    }

    std::cout
        << "Dispositivo de entrada:\n"
        << "  " << deviceInfo->name << '\n'
        << "  Canales: "
        << deviceInfo->maxInputChannels
        << '\n'
        << "  Sample rate: "
        << deviceInfo->defaultSampleRate
        << " Hz\n\n";

    AudioData audioData;

    PaStream* stream = nullptr;

    error = Pa_OpenDefaultStream(
        &stream,
        CHANNELS,
        0,
        paFloat32,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        audioCallback,
        &audioData
    );

    if (error != paNoError)
    {
        std::cerr
            << "Error abriendo el stream: "
            << Pa_GetErrorText(error)
            << '\n';

        Pa_Terminate();

        return 1;
    }

    error = Pa_StartStream(stream);

    if (error != paNoError)
    {
        std::cerr
            << "Error iniciando el stream: "
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
        std::cerr
            << "Error inicializando GLFW.\n";

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MAJOR,
        3
    );

    glfwWindowHint(
        GLFW_CONTEXT_VERSION_MINOR,
        3
    );

    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );

    // ========================================================
    // PANTALLA COMPLETA
    // ========================================================

    GLFWmonitor* monitor =
        glfwGetPrimaryMonitor();

    if (monitor == nullptr)
    {
        std::cerr
            << "No se encontro el monitor principal.\n";

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
        std::cerr
            << "No se pudo obtener la resolucion del monitor.\n";

        glfwTerminate();

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }

    GLFWwindow* window =
        glfwCreateWindow(
            videoMode->width,
            videoMode->height,
            "AUDIO",
            monitor,
            nullptr
        );

    if (window == nullptr)
    {
        std::cerr
            << "Error creando la ventana OpenGL.\n";

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
        std::cerr
            << "Error cargando GLAD.\n";

        glfwDestroyWindow(window);
        glfwTerminate();

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }

    std::cout
        << "OpenGL inicializado correctamente.\n"
        << "Version: "
        << glGetString(GL_VERSION)
        << '\n'
        << "Renderer: "
        << glGetString(GL_RENDERER)
        << '\n'
        << "Resolucion: "
        << videoMode->width
        << " x "
        << videoMode->height
        << "\n\n";

    // ========================================================
    // SHADERS
    // ========================================================

    const GLuint shaderProgram =
        createShaderProgram();

    if (shaderProgram == 0)
    {
        glfwDestroyWindow(window);
        glfwTerminate();

        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        Pa_Terminate();

        return 1;
    }

    // ========================================================
    // WAVEFORM VAO / VBO
    // ========================================================

    GLuint VAO = 0;
    GLuint VBO = 0;

    glGenVertexArrays(
        1,
        &VAO
    );

    glGenBuffers(
        1,
        &VBO
    );

    glBindVertexArray(VAO);

    glBindBuffer(
        GL_ARRAY_BUFFER,
        VBO
    );

    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) *
            2 *
            WAVEFORM_SAMPLES,
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

    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );

    glBindVertexArray(0);

    // ========================================================
    // MAIN LOOP
    // ========================================================

    while (!glfwWindowShouldClose(window))
    {
        // ----------------------------------------------------
        // ESC
        // ----------------------------------------------------

        if (glfwGetKey(window, GLFW_KEY_ESCAPE)
            == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(
                window,
                GLFW_TRUE
            );
        }

        // ----------------------------------------------------
        // Obtener indice actual
        // ----------------------------------------------------

        const unsigned int writeIndex =
            audioData.writeIndex.load(
                std::memory_order_relaxed
            );

        // ----------------------------------------------------
        // Construir waveform
        // ----------------------------------------------------

        std::array<float, WAVEFORM_SAMPLES * 2>
            vertices;

        for (int i = 0;
             i < WAVEFORM_SAMPLES;
             ++i)
        {
            const unsigned int sampleIndex =
                (writeIndex + i)
                % WAVEFORM_SAMPLES;

            const float sample =
                audioData.samples[sampleIndex].load(
                    std::memory_order_relaxed
                );

            const float x =
                -1.0f +
                2.0f *
                static_cast<float>(i) /
                static_cast<float>(
                    WAVEFORM_SAMPLES - 1
                );

            const float y =
                std::clamp(
                    sample * WAVEFORM_SCALE,
                    -1.0f,
                    1.0f
                );

            vertices[i * 2] = x;
            vertices[i * 2 + 1] = y;
        }

        // ----------------------------------------------------
        // Actualizar VBO
        // ----------------------------------------------------

        glBindBuffer(
            GL_ARRAY_BUFFER,
            VBO
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

        // ----------------------------------------------------
        // Render
        // ----------------------------------------------------

        glClearColor(
            0.03f,
            0.03f,
            0.03f,
            1.0f
        );

        glClear(
            GL_COLOR_BUFFER_BIT
        );

        glUseProgram(
            shaderProgram
        );

        glBindVertexArray(VAO);

        glLineWidth(2.0f);

        glDrawArrays(
            GL_LINE_STRIP,
            0,
            WAVEFORM_SAMPLES
        );

        glBindVertexArray(0);

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    // ========================================================
    // CLEANUP OPENGL
    // ========================================================

    glDeleteVertexArrays(
        1,
        &VAO
    );

    glDeleteBuffers(
        1,
        &VBO
    );

    glDeleteProgram(
        shaderProgram
    );

    // ========================================================
    // CLEANUP GLFW
    // ========================================================

    glfwDestroyWindow(window);
    glfwTerminate();

    // ========================================================
    // CLEANUP PORTAUDIO
    // ========================================================

    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout
        << "AUDIO finalizado.\n";

    return 0;
}