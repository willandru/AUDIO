#include <cmath>
#include <iostream>

#include <portaudio.h>

constexpr int SAMPLE_RATE = 44100;
constexpr int FRAMES_PER_BUFFER = 256;
constexpr int CHANNELS = 1;

double calculateRMS(const float* samples, unsigned long count)
{
    if (samples == nullptr || count == 0)
    {
        return 0.0;
    }

    double sumSquares = 0.0;

    for (unsigned long i = 0; i < count; ++i)
    {
        sumSquares += static_cast<double>(samples[i]) *
                      static_cast<double>(samples[i]);
    }

    return std::sqrt(sumSquares / static_cast<double>(count));
}

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
    (void)userData;

    const float* input =
        static_cast<const float*>(inputBuffer);

    if (input == nullptr)
    {
        return paContinue;
    }

    double rms = calculateRMS(input, framesPerBuffer);

    static int counter = 0;

    ++counter;

    if (counter % 20 == 0)
    {
        std::cout << "RMS: " << rms << '\n';
    }

    return paContinue;
}

int main()
{
    PaError error = Pa_Initialize();

    if (error != paNoError)
    {
        std::cerr << "Error inicializando PortAudio: "
                  << Pa_GetErrorText(error) << '\n';

        return 1;
    }

    int deviceIndex = Pa_GetDefaultInputDevice();

    if (deviceIndex == paNoDevice)
    {
        std::cerr << "No se encontro un dispositivo de entrada.\n";

        Pa_Terminate();
        return 1;
    }

    const PaDeviceInfo* deviceInfo =
        Pa_GetDeviceInfo(deviceIndex);

    if (deviceInfo == nullptr)
    {
        std::cerr << "No se pudo obtener informacion del dispositivo.\n";

        Pa_Terminate();
        return 1;
    }

    std::cout << "Dispositivo de entrada:\n";
    std::cout << "  " << deviceInfo->name << '\n';
    std::cout << "  Canales: "
              << deviceInfo->maxInputChannels << '\n';
    std::cout << "  Sample rate: "
              << deviceInfo->defaultSampleRate << " Hz\n\n";

    PaStream* stream = nullptr;

    error = Pa_OpenDefaultStream(
        &stream,
        CHANNELS,
        0,
        paFloat32,
        SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        audioCallback,
        nullptr
    );

    if (error != paNoError)
    {
        std::cerr << "Error abriendo el stream: "
                  << Pa_GetErrorText(error) << '\n';

        Pa_Terminate();
        return 1;
    }

    error = Pa_StartStream(stream);

    if (error != paNoError)
    {
        std::cerr << "Error iniciando el stream: "
                  << Pa_GetErrorText(error) << '\n';

        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    std::cout << "Capturando audio...\n";
    std::cout << "Presiona ENTER para detener.\n\n";

    std::cin.get();

    error = Pa_StopStream(stream);

    if (error != paNoError)
    {
        std::cerr << "Error deteniendo el stream: "
                  << Pa_GetErrorText(error) << '\n';
    }

    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "\nCaptura finalizada.\n";

    return 0;
}