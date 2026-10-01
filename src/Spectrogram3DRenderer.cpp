#include "Spectrogram3DRenderer.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
    constexpr float WIDTH_SCALE = 8.0f;
    constexpr float DEPTH_SCALE = 8.0f;
    constexpr float HEIGHT_SCALE = 3.0f;

    constexpr int COLUMN_STEP = 4;
    constexpr int BIN_STEP = 4;

    constexpr float GRAPH_TILT_DEGREES = 30.0f;
    constexpr float GRAPH_Y_OFFSET = 1.0f;


    struct Matrix4
    {
        float value[16]{};
    };


    Matrix4 identityMatrix()
    {
        Matrix4 result{};

        result.value[0] = 1.0f;
        result.value[5] = 1.0f;
        result.value[10] = 1.0f;
        result.value[15] = 1.0f;

        return result;
    }


    Matrix4 perspectiveMatrix(
        float fovRadians,
        float aspect,
        float nearPlane,
        float farPlane
    )
    {
        Matrix4 result{};

        const float tangent =
            std::tan(
                fovRadians * 0.5f
            );

        result.value[0] =
            1.0f /
            (aspect * tangent);

        result.value[5] =
            1.0f /
            tangent;

        result.value[10] =
            -(
                farPlane +
                nearPlane
            ) /
            (
                farPlane -
                nearPlane
            );

        result.value[11] = -1.0f;

        result.value[14] =
            -(
                2.0f *
                farPlane *
                nearPlane
            ) /
            (
                farPlane -
                nearPlane
            );

        return result;
    }


    Matrix4 lookAtMatrix(
        float eyeX,
        float eyeY,
        float eyeZ,
        float centerX,
        float centerY,
        float centerZ
    )
    {
        const float forwardX =
            centerX - eyeX;

        const float forwardY =
            centerY - eyeY;

        const float forwardZ =
            centerZ - eyeZ;

        const float forwardLength =
            std::sqrt(
                forwardX * forwardX +
                forwardY * forwardY +
                forwardZ * forwardZ
            );

        const float fx =
            forwardX / forwardLength;

        const float fy =
            forwardY / forwardLength;

        const float fz =
            forwardZ / forwardLength;


        const float upX = 0.0f;
        const float upY = 1.0f;
        const float upZ = 0.0f;


        float sideX =
            fy * upZ -
            fz * upY;

        float sideY =
            fz * upX -
            fx * upZ;

        float sideZ =
            fx * upY -
            fy * upX;


        const float sideLength =
            std::sqrt(
                sideX * sideX +
                sideY * sideY +
                sideZ * sideZ
            );


        sideX /= sideLength;
        sideY /= sideLength;
        sideZ /= sideLength;


        const float realUpX =
            sideY * fz -
            sideZ * fy;

        const float realUpY =
            sideZ * fx -
            sideX * fz;

        const float realUpZ =
            sideX * fy -
            sideY * fx;


        Matrix4 result =
            identityMatrix();


        result.value[0] = sideX;
        result.value[1] = realUpX;
        result.value[2] = -fx;

        result.value[4] = sideY;
        result.value[5] = realUpY;
        result.value[6] = -fy;

        result.value[8] = sideZ;
        result.value[9] = realUpZ;
        result.value[10] = -fz;


        result.value[12] =
            -(
                sideX * eyeX +
                sideY * eyeY +
                sideZ * eyeZ
            );


        result.value[13] =
            -(
                realUpX * eyeX +
                realUpY * eyeY +
                realUpZ * eyeZ
            );


        result.value[14] =
            fx * eyeX +
            fy * eyeY +
            fz * eyeZ;


        return result;
    }


    Matrix4 rotationXMatrix(
        float angleRadians
    )
    {
        Matrix4 result =
            identityMatrix();

        const float cosine =
            std::cos(angleRadians);

        const float sine =
            std::sin(angleRadians);

        result.value[5] = cosine;
        result.value[6] = sine;

        result.value[9] = -sine;
        result.value[10] = cosine;

        return result;
    }


    Matrix4 translationMatrix(
        float x,
        float y,
        float z
    )
    {
        Matrix4 result =
            identityMatrix();

        result.value[12] = x;
        result.value[13] = y;
        result.value[14] = z;

        return result;
    }


    Matrix4 multiplyMatrix(
        const Matrix4& a,
        const Matrix4& b
    )
    {
        Matrix4 result{};

        for (int column = 0; column < 4; ++column)
        {
            for (int row = 0; row < 4; ++row)
            {
                result.value[
                    column * 4 + row
                ] =
                    a.value[0 * 4 + row] *
                    b.value[column * 4 + 0] +

                    a.value[1 * 4 + row] *
                    b.value[column * 4 + 1] +

                    a.value[2 * 4 + row] *
                    b.value[column * 4 + 2] +

                    a.value[3 * 4 + row] *
                    b.value[column * 4 + 3];
            }
        }

        return result;
    }
}


// ============================================================
// CONSTRUCTOR
// ============================================================

Spectrogram3DRenderer::Spectrogram3DRenderer()
{
}


// ============================================================
// DESTRUCTOR
// ============================================================

Spectrogram3DRenderer::~Spectrogram3DRenderer()
{
    destroy();
}


// ============================================================
// INICIALIZAR
// ============================================================

bool Spectrogram3DRenderer::initialize(
    int newWidth,
    int newHeight
)
{
    if (
        newWidth < 2 ||
        newHeight < 2
    )
    {
        return false;
    }


    destroy();


    width = newWidth;
    height = newHeight;


    meshWidth =
        (
            width - 1
        ) /
        COLUMN_STEP +
        1;


    meshHeight =
        (
            height - 1
        ) /
        BIN_STEP +
        1;


    createMesh();


    return
        vao != 0 &&
        vbo != 0 &&
        ebo != 0;
}


// ============================================================
// CREAR MALLA
// ============================================================

void Spectrogram3DRenderer::createMesh()
{
    std::vector<Vertex> vertices;

    std::vector<unsigned int> indices;


    vertices.resize(
        static_cast<std::size_t>(
            meshWidth
        ) *
        static_cast<std::size_t>(
            meshHeight
        )
    );


    indices.reserve(
        static_cast<std::size_t>(
            meshWidth - 1
        ) *
        static_cast<std::size_t>(
            meshHeight - 1
        ) *
        6
    );


    // ========================================================
    // VÉRTICES
    // ========================================================

    for (
        int z = 0;
        z < meshHeight;
        ++z
    )
    {
        const int sourceBin =
            std::min(
                z * BIN_STEP,
                height - 1
            );


        const float normalizedZ =
            static_cast<float>(
                sourceBin
            ) /
            static_cast<float>(
                height - 1
            );


        const float positionZ =
            (
                0.5f -
                normalizedZ
            ) *
            DEPTH_SCALE;


        for (
            int x = 0;
            x < meshWidth;
            ++x
        )
        {
            const int sourceColumn =
                std::min(
                    x * COLUMN_STEP,
                    width - 1
                );


            const float normalizedX =
                static_cast<float>(
                    sourceColumn
                ) /
                static_cast<float>(
                    width - 1
                );


            const float positionX =
                (
                    normalizedX -
                    0.5f
                ) *
                WIDTH_SCALE;


            const int index =
                z * meshWidth + x;


            vertices[index] =
            {
                positionX,
                0.0f,
                positionZ
            };
        }
    }


    // ========================================================
    // ÍNDICES
    // ========================================================

    for (
        int z = 0;
        z < meshHeight - 1;
        ++z
    )
    {
        for (
            int x = 0;
            x < meshWidth - 1;
            ++x
        )
        {
            const unsigned int topLeft =
                static_cast<unsigned int>(
                    z * meshWidth + x
                );


            const unsigned int topRight =
                static_cast<unsigned int>(
                    z * meshWidth +
                    x +
                    1
                );


            const unsigned int bottomLeft =
                static_cast<unsigned int>(
                    (z + 1) *
                    meshWidth +
                    x
                );


            const unsigned int bottomRight =
                static_cast<unsigned int>(
                    (z + 1) *
                    meshWidth +
                    x +
                    1
                );


            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);


            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }


    // ========================================================
    // CONTADORES
    // ========================================================

    vertexCount =
        static_cast<int>(
            vertices.size()
        );


    indexCount =
        static_cast<int>(
            indices.size()
        );


    // ========================================================
    // OPENGL
    // ========================================================

    glGenVertexArrays(
        1,
        &vao
    );


    glGenBuffers(
        1,
        &vbo
    );


    glGenBuffers(
        1,
        &ebo
    );


    glBindVertexArray(
        vao
    );


    glBindBuffer(
        GL_ARRAY_BUFFER,
        vbo
    );


    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(
            vertices.size() *
            sizeof(Vertex)
        ),
        vertices.data(),
        GL_DYNAMIC_DRAW
    );


    glBindBuffer(
        GL_ELEMENT_ARRAY_BUFFER,
        ebo
    );


    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(
            indices.size() *
            sizeof(unsigned int)
        ),
        indices.data(),
        GL_STATIC_DRAW
    );


    glEnableVertexAttribArray(0);


    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        sizeof(Vertex),
        nullptr
    );


    glBindVertexArray(0);
}


// ============================================================
// ACTUALIZAR
// ============================================================

void Spectrogram3DRenderer::update(
    const unsigned char* data,
    int dataWidth,
    int dataHeight,
    int currentColumn
)
{
    if (data == nullptr)
    {
        return;
    }


    if (
        dataWidth != width ||
        dataHeight != height
    )
    {
        return;
    }


    std::vector<Vertex> vertices;


    vertices.resize(
        static_cast<std::size_t>(
            meshWidth
        ) *
        static_cast<std::size_t>(
            meshHeight
        )
    );


    // ========================================================
    // NORMALIZAR COLUMNA ACTUAL
    // ========================================================

    currentColumn =
        (
            currentColumn %
            width +
            width
        ) %
        width;


    // ========================================================
    // VÉRTICES
    // ========================================================

    for (
        int z = 0;
        z < meshHeight;
        ++z
    )
    {
        const int sourceBin =
            std::min(
                z * BIN_STEP,
                height - 1
            );


        const float normalizedZ =
            static_cast<float>(
                sourceBin
            ) /
            static_cast<float>(
                height - 1
            );


        const float positionZ =
            (
                0.5f -
                normalizedZ
            ) *
            DEPTH_SCALE;


        for (
            int x = 0;
            x < meshWidth;
            ++x
        )
        {
            const int temporalColumn =
                std::min(
                    x * COLUMN_STEP,
                    width - 1
                );


            const int sourceColumn =
                (
                    currentColumn +
                    temporalColumn
                ) %
                width;


            const float normalizedX =
                static_cast<float>(
                    temporalColumn
                ) /
                static_cast<float>(
                    width - 1
                );


            const float positionX =
                (
                    normalizedX -
                    0.5f
                ) *
                WIDTH_SCALE;


            const int dataIndex =
                sourceBin *
                width +
                sourceColumn;


            const float normalizedMagnitude =
                static_cast<float>(
                    data[dataIndex]
                ) /
                255.0f;


            const float positionY =
                normalizedMagnitude *
                HEIGHT_SCALE;


            const int vertexIndex =
                z * meshWidth + x;


            vertices[vertexIndex] =
            {
                positionX,
                positionY,
                positionZ
            };
        }
    }


    // ========================================================
    // ACTUALIZAR VBO
    // ========================================================

    glBindBuffer(
        GL_ARRAY_BUFFER,
        vbo
    );


    glBufferSubData(
        GL_ARRAY_BUFFER,
        0,
        static_cast<GLsizeiptr>(
            vertices.size() *
            sizeof(Vertex)
        ),
        vertices.data()
    );


    glBindBuffer(
        GL_ARRAY_BUFFER,
        0
    );
}


// ============================================================
// RENDER
// ============================================================

void Spectrogram3DRenderer::render(
    GLuint shader
) const
{
    if (
        vao == 0 ||
        indexCount == 0
    )
    {
        return;
    }


    glUseProgram(
        shader
    );


    const Matrix4 cameraView =
        lookAtMatrix(
            0.0f,
            5.0f,
            9.0f,
            0.0f,
            0.7f,
            0.0f
        );


    const float tiltRadians =
        GRAPH_TILT_DEGREES *
        3.14159265358979323846f /
        180.0f;


    const Matrix4 rotation =
        rotationXMatrix(
            tiltRadians
        );


    const Matrix4 translation =
        translationMatrix(
            0.0f,
            GRAPH_Y_OFFSET,
            0.0f
        );


    const Matrix4 model =
        multiplyMatrix(
            translation,
            rotation
        );


    const Matrix4 view =
        multiplyMatrix(
            cameraView,
            model
        );


    const Matrix4 projection =
        perspectiveMatrix(
            45.0f *
                3.14159265358979323846f /
                180.0f,
            1.6f,
            0.1f,
            100.0f
        );


    const GLint viewLocation =
        glGetUniformLocation(
            shader,
            "view"
        );


    const GLint projectionLocation =
        glGetUniformLocation(
            shader,
            "projection"
        );


    if (viewLocation >= 0)
    {
        glUniformMatrix4fv(
            viewLocation,
            1,
            GL_FALSE,
            view.value
        );
    }


    if (projectionLocation >= 0)
    {
        glUniformMatrix4fv(
            projectionLocation,
            1,
            GL_FALSE,
            projection.value
        );
    }


    glBindVertexArray(
        vao
    );


    glDrawElements(
        GL_TRIANGLES,
        indexCount,
        GL_UNSIGNED_INT,
        nullptr
    );


    glBindVertexArray(0);
}


// ============================================================
// DESTRUIR
// ============================================================

void Spectrogram3DRenderer::destroy()
{
    if (ebo != 0)
    {
        glDeleteBuffers(
            1,
            &ebo
        );

        ebo = 0;
    }


    if (vbo != 0)
    {
        glDeleteBuffers(
            1,
            &vbo
        );

        vbo = 0;
    }


    if (vao != 0)
    {
        glDeleteVertexArrays(
            1,
            &vao
        );

        vao = 0;
    }


    vertexCount = 0;

    indexCount = 0;

    meshWidth = 0;

    meshHeight = 0;

    width = 0;

    height = 0;
}