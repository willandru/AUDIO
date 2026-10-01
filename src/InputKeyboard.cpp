#include "InputKeyboard.h"

// ============================================================
// PROCESAR TECLADO
// ============================================================

void InputKeyboard::process(GLFWwindow* window)
{
    previousLeftPressed = leftPressed;
    previousRightPressed = rightPressed;

    previousAPressed = aPressed;
    previousDPressed = dPressed;


    leftPressed =
        glfwGetKey(
            window,
            GLFW_KEY_LEFT
        ) == GLFW_PRESS;


    rightPressed =
        glfwGetKey(
            window,
            GLFW_KEY_RIGHT
        ) == GLFW_PRESS;


    aPressed =
        glfwGetKey(
            window,
            GLFW_KEY_A
        ) == GLFW_PRESS;


    dPressed =
        glfwGetKey(
            window,
            GLFW_KEY_D
        ) == GLFW_PRESS;


    // ========================================================
    // ESC
    // ========================================================

    if (
        glfwGetKey(
            window,
            GLFW_KEY_ESCAPE
        ) == GLFW_PRESS
    )
    {
        glfwSetWindowShouldClose(
            window,
            GLFW_TRUE
        );
    }
}


// ============================================================
// FLECHA IZQUIERDA
// ============================================================

bool InputKeyboard::isLeftPressed() const
{
    return
        leftPressed &&
        !previousLeftPressed;
}


// ============================================================
// FLECHA DERECHA
// ============================================================

bool InputKeyboard::isRightPressed() const
{
    return
        rightPressed &&
        !previousRightPressed;
}


// ============================================================
// TECLA A
// ============================================================

bool InputKeyboard::isAPressed() const
{
    return
        aPressed &&
        !previousAPressed;
}


// ============================================================
// TECLA D
// ============================================================

bool InputKeyboard::isDPressed() const
{
    return
        dPressed &&
        !previousDPressed;
}