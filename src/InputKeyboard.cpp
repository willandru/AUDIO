#include "InputKeyboard.h"

// ============================================================
// PROCESAR TECLADO
// ============================================================

void InputKeyboard::process(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(
            window,
            GLFW_TRUE
        );
    }
}