#version 330 core

in float intensity;

out vec4 FragColor;

void main()
{
    vec3 low =
        vec3(
            0.01,
            0.02,
            0.08
        );

    vec3 middle =
        vec3(
            0.0,
            0.55,
            0.85
        );

    vec3 high =
        vec3(
            1.0,
            0.85,
            0.15
        );

    vec3 peak =
        vec3(
            1.0,
            0.05,
            0.0
        );

    float adjustedIntensity =
        pow(
            clamp(intensity, 0.0, 1.0),
            0.35
        );

    vec3 color;

    if (adjustedIntensity < 0.5)
    {
        color =
            mix(
                low,
                middle,
                adjustedIntensity * 2.0
            );
    }
    else if (adjustedIntensity < 0.8)
    {
        color =
            mix(
                middle,
                high,
                (adjustedIntensity - 0.5) / 0.3
            );
    }
    else
    {
        color =
            mix(
                high,
                peak,
                (adjustedIntensity - 0.8) / 0.2
            );
    }

    FragColor =
        vec4(
            color,
            1.0
        );
}