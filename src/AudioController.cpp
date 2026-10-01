#include "AudioController.h"

#include <cmath>
#include <iostream>
#include <limits>

#include <pa_win_wasapi.h>


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

    PaHostApiIndex wasapiHostApi =
        Pa_HostApiTypeIdToHostApiIndex(paWASAPI);

    if (wasapiHostApi == paHostApiNotFound)
    {
        std::cerr
            << "WASAPI no está disponible en PortAudio.\n";

        return false;
    }

    const PaHostApiInfo* hostApiInfo =
        Pa_GetHostApiInfo(wasapiHostApi);

    if (hostApiInfo == nullptr)
    {
        std::cerr
            << "No se pudo obtener información de WASAPI.\n";

        return false;
    }

    PaDeviceIndex inputDevice = paNoDevice;

    for (int i = 0;
         i < hostApiInfo->deviceCount;
         ++i)
    {
        const PaDeviceIndex device =
            Pa_HostApiDeviceIndexToDeviceIndex(
                wasapiHostApi,
                i
            );

        if (device == paNoDevice)
        {
            continue;
        }

        const PaDeviceInfo* deviceInfo =
            Pa_GetDeviceInfo(device);

        if (deviceInfo == nullptr)
        {
            continue;
        }

        if (deviceInfo->maxInputChannels > 0)
        {
            inputDevice = device;
            break;
        }
    }

    if (inputDevice == paNoDevice)
    {
        std::cerr
            << "No se encontró dispositivo de entrada WASAPI.\n";

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
        << "Host API: "
        << hostApiInfo->name
        << '\n'
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
    // CONFIGURACIÓN WASAPI
    // ========================================================

    PaWasapiStreamInfo wasapiStreamInfo{};

    wasapiStreamInfo.size =
        sizeof(PaWasapiStreamInfo);

    wasapiStreamInfo.hostApiType =
        paWASAPI;

    wasapiStreamInfo.version = 1;

    wasapiStreamInfo.flags = 0;

    wasapiStreamInfo.streamOption =
        eStreamOptionRaw;


    // ========================================================
    // PARÁMETROS DE ENTRADA
    // ========================================================

    PaStreamParameters inputParameters{};

    inputParameters.device =
        inputDevice;

    inputParameters.channelCount =
        CHANNELS;

    inputParameters.sampleFormat =
        paFloat32;

    inputParameters.suggestedLatency =
        deviceInfo->defaultLowInputLatency;

    inputParameters.hostApiSpecificStreamInfo =
        &wasapiStreamInfo;


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


    // ========================================================
    // DIAGNÓSTICO DE LA SEÑAL RECIBIDA
    // ========================================================

    static double accumulatedSquared = 0.0;

    static float minimum =
        std::numeric_limits<float>::max();

    static float maximum =
        std::numeric_limits<float>::lowest();

    static unsigned long accumulatedSamples = 0;

    static int callbackCount = 0;

    for (unsigned long i = 0;
         i < frameCount;
         ++i)
    {
        const float sample =
            inputSamples[i];

        accumulatedSquared +=
            static_cast<double>(sample) *
            static_cast<double>(sample);

        if (sample < minimum)
        {
            minimum = sample;
        }

        if (sample > maximum)
        {
            maximum = sample;
        }

        ++accumulatedSamples;


        // ====================================================
        // GUARDAR MUESTRA
        // ====================================================

        const int index =
            audioData->writeIndex.fetch_add(
                1,
                std::memory_order_relaxed
            ) % WAVEFORM_SAMPLES;

        audioData->samples[index].store(
            sample,
            std::memory_order_relaxed
        );
    }


    // ========================================================
    // REPORTAR APROXIMADAMENTE CADA SEGUNDO
    // ========================================================

    ++callbackCount;

    if (callbackCount >=
        SAMPLE_RATE / FRAMES_PER_BUFFER)
    {
        const double rms =
            accumulatedSamples > 0
                ? std::sqrt(
                    accumulatedSquared /
                    static_cast<double>(
                        accumulatedSamples
                    )
                )
                : 0.0;

        std::cout
            << "Audio recibido | "
            << "RMS: "
            << rms
            << " | Min: "
            << minimum
            << " | Max: "
            << maximum
            << '\n';

        accumulatedSquared = 0.0;

        minimum =
            std::numeric_limits<float>::max();

        maximum =
            std::numeric_limits<float>::lowest();

        accumulatedSamples = 0;

        callbackCount = 0;
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

    for (int i = 0;
         i < sampleCount;
         ++i)
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