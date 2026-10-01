#version 330 core

layout(location = 0) in vec3 aPosition;

uniform mat4 view;
uniform mat4 projection;

out float intensity;

void main()
{
    intensity =
        clamp(
            aPosition.y / 3.0,
            0.0,
            1.0
        );

    gl_Position =
        projection *
        view *
        vec4(
            aPosition,
            1.0
        );
}