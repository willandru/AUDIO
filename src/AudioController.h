#pragma once

#include <array>
#include <atomic>

#include <portaudio.h>

// ============================================================
// CONFIGURACIÓN DE AUDIO
// ============================================================

constexpr int SAMPLE_RATE = 44100;
constexpr int FRAMES_PER_BUFFER = 256;
constexpr int CHANNELS = 1;

constexpr int WAVEFORM_SAMPLES = 1024;


// ============================================================
// DATOS DE AUDIO
// ============================================================

struct AudioData
{
    std::array<
        std::atomic<float>,
        WAVEFORM_SAMPLES
    > samples{};

    std::atomic<int> writeIndex{ 0 };
};


// ============================================================
// CONTROLADOR DE AUDIO
// ============================================================

class AudioController
{
public:

    AudioController();
    ~AudioController();

    bool initialize();
    bool start();
    void stop();

    void copyLatestSamples(
        float* destination,
        int sampleCount
    ) const;

    const AudioData& getAudioData() const;

private:

    static int audioCallback(
        const void* input,
        void* output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags,
        void* userData
    );

    AudioData audioData;
    PaStream* stream = nullptr;

    bool initialized = false;
    bool running = false;
};