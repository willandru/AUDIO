#version 330 core

in vec2 TexCoord;

uniform sampler2D uSpectrum;
uniform float uColumn;

out vec4 FragColor;

void main()
{
    float normalizedColumn = uColumn / 1024.0;

    float x = fract(
        TexCoord.x + normalizedColumn
    );

    float value = texture(
        uSpectrum,
        vec2(x, TexCoord.y)
    ).r;

    float enhanced = pow(value, 0.55);

    vec3 color;

    if (enhanced < 0.25)
    {
        float t = enhanced / 0.25;

        color = mix(
            vec3(0.0, 0.0, 0.015),
            vec3(0.0, 0.15, 0.8),
            t
        );
    }
    else if (enhanced < 0.50)
    {
        float t = (enhanced - 0.25) / 0.25;

        color = mix(
            vec3(0.0, 0.15, 0.8),
            vec3(0.0, 0.9, 1.0),
            t
        );
    }
    else if (enhanced < 0.72)
    {
        float t = (enhanced - 0.50) / 0.22;

        color = mix(
            vec3(0.0, 0.9, 1.0),
            vec3(1.0, 1.0, 0.0),
            t
        );
    }
    else
    {
        float t = (enhanced - 0.72) / 0.28;

        color = mix(
            vec3(1.0, 1.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            t
        );
    }

    FragColor = vec4(color, 1.0);
}