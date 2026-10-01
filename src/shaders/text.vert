#version 330 core

layout (location = 0) in vec4 vertex;

uniform vec2 uScreenSize;

out vec2 TexCoord;

void main()
{
    vec2 position =
        vertex.xy;

    vec2 normalized =
        vec2(
            position.x /
                uScreenSize.x *
                2.0 -
                1.0,

            1.0 -
                position.y /
                uScreenSize.y *
                2.0
        );

    gl_Position =
        vec4(
            normalized,
            0.0,
            1.0
        );

    TexCoord =
        vertex.zw;
}