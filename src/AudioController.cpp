#include "AudioController.h"

#include <iostream>

// ============================================================
// CONSTRUCTOR
// ============================================================

AudioController::AudioController()
{
}


// ============================================================
// DESTRUCTOR
// ============================================================

AudioController::~AudioController()
{
    stop();
}


// ============================================================
// INICIALIZAR PORTAUDIO
// ============================================================

bool AudioController::initialize()
{
    if (initialized)
    {
        return true;
    }

    const PaError error = Pa_Initialize();

    if (error != paNoError)
    {
        std::cerr
            << "Error inicializando PortAudio: "
            << Pa_GetErrorText(error)
            << '\n';

        return false;
    }

    initialized = true;

    return true;
}


// ============================================================
// INICIAR CAPTURA
// ============================================================

bool AudioController::start()
{
    if (!initialized)
    {
        return false;
    }

    if (running)
    {
        return true;
    }

    const PaDeviceIndex inputDevice =
        Pa_GetDefaultInputDevice();

    if (inputDevice == paNoDevice)
    {
        std::cerr
            << "No se encontró dispositivo de entrada.\n";

        return false;
    }

    const PaDeviceInfo* deviceInfo =
        Pa_GetDeviceInfo(inputDevice);

    if (deviceInfo == nullptr)
    {
        std::cerr
            << "No se pudo obtener información del dispositivo de entrada.\n";

        return false;
    }

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


    // ========================================================
    // PARÁMETROS DE ENTRADA
    // ========================================================

    PaStreamParameters inputParameters{};

    inputParameters.device = inputDevice;

    inputParameters.channelCount = CHANNELS;

    inputParameters.sampleFormat = paFloat32;

    inputParameters.suggestedLatency =
        deviceInfo->defaultLowInputLatency;

    inputParameters.hostApiSpecificStreamInfo =
        nullptr;


    // ========================================================
    // ABRIR STREAM
    // ========================================================

    PaError error = Pa_OpenStream(
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

        stream = nullptr;

        return false;
    }


    // ========================================================
    // INICIAR STREAM
    // ========================================================

    error = Pa_StartStream(stream);

    if (error != paNoError)
    {
        std::cerr
            << "Error iniciando stream: "
            << Pa_GetErrorText(error)
            << '\n';

        Pa_CloseStream(stream);

        stream = nullptr;

        return false;
    }

    running = true;

    return true;
}


// ============================================================
// DETENER AUDIO
// ============================================================

void AudioController::stop()
{
    if (running && stream != nullptr)
    {
        Pa_StopStream(stream);

        running = false;
    }

    if (stream != nullptr)
    {
        Pa_CloseStream(stream);

        stream = nullptr;
    }

    if (initialized)
    {
        Pa_Terminate();

        initialized = false;
    }
}


// ============================================================
// CALLBACK DE AUDIO
// ============================================================

int AudioController::audioCallback(
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
// OBTENER ÚLTIMAS MUESTRAS
// ============================================================

void AudioController::copyLatestSamples(
    float* destination,
    int sampleCount) const
{
    const int currentWriteIndex =
        audioData.writeIndex.load(
            std::memory_order_relaxed
        );

    for (int i = 0; i < sampleCount; ++i)
    {
        const int index =
            (
                (
                    currentWriteIndex -
                    sampleCount +
                    i
                ) %
                WAVEFORM_SAMPLES +
                WAVEFORM_SAMPLES
            ) %
            WAVEFORM_SAMPLES;

        destination[i] =
            audioData.samples[index].load(
                std::memory_order_relaxed
            );
    }
}


// ============================================================
// ACCESO A LOS DATOS
// ============================================================

const AudioData& AudioController::getAudioData() const
{
    return audioData;
}