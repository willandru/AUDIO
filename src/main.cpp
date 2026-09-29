
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>

#include <portaudio.h>

constexpr int SAMPLE_RATE = 44100;
constexpr int FRAMES_PER_BUFFER = 256;
constexpr int CHANNELS = 1;

constexpr int BAR_WIDTH = 50;
constexpr double RMS_SCALE = 0.05;

struct AudioData
{
    std::atomic<double> rms{0.0};
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

    auto* audioData = static_cast<AudioData*>(userData);
    const auto* samples = static_cast<const float*>(inputBuffer);

    if (samples == nullptr || framesPerBuffer == 0)
    {
        audioData->rms.store(0.0);
        return paContinue;
    }

    double sumSquares = 0.0;

    for (unsigned long i = 0; i < framesPerBuffer; ++i)
    {
        const double sample = samples[i];
        sumSquares += sample * sample;
    }

    const double rms = std::sqrt(
        sumSquares / static_cast<double>(framesPerBuffer));

    audioData->rms.store(rms);

    return paContinue;
}

void printAudioLevel(double rms)
{
    const double normalized = std::clamp(
        rms / RMS_SCALE, 0.0, 1.0);

    const int barLength = static_cast<int>(
        normalized * BAR_WIDTH);

    std::cout << '\r' << "RMS: " << rms << " | ";

    for (int i = 0; i < BAR_WIDTH; ++i)
    {
        std::cout << (i < barLength ? '#' : ' ');
    }

    std::cout << std::flush;
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

    const PaDeviceIndex deviceIndex = Pa_GetDefaultInputDevice();

    if (deviceIndex == paNoDevice)
    {
        std::cerr << "No se encontro un dispositivo de entrada.\n";
        Pa_Terminate();
        return 1;
    }

    const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(deviceIndex);

    if (deviceInfo == nullptr)
    {
        std::cerr << "No se pudo obtener informacion del dispositivo.\n";
        Pa_Terminate();
        return 1;
    }

    std::cout << "Dispositivo de entrada:\n"
              << "  " << deviceInfo->name << '\n'
              << "  Canales: " << deviceInfo->maxInputChannels << '\n'
              << "  Sample rate: " << deviceInfo->defaultSampleRate
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

    std::atomic<bool> running{true};

    std::thread displayThread([&]()
    {
        while (running.load())
        {
            printAudioLevel(audioData.rms.load());
            std::this_thread::sleep_for(
                std::chrono::milliseconds(50));
        }
    });

    std::cin.get();

    running.store(false);
    displayThread.join();

    error = Pa_StopStream(stream);

    if (error != paNoError)
    {
        std::cerr << "\nError deteniendo el stream: "
                  << Pa_GetErrorText(error) << '\n';
    }

    Pa_CloseStream(stream);
    Pa_Terminate();

    std::cout << "\n\nCaptura finalizada.\n";

    return 0;
}