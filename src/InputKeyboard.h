#pragma once

#include <GLFW/glfw3.h>

class InputKeyboard
{
public:

    void process(GLFWwindow* window);

    bool isLeftPressed() const;
    bool isRightPressed() const;

    bool isAPressed() const;
    bool isDPressed() const;

private:

    bool leftPressed = false;
    bool rightPressed = false;

    bool aPressed = false;
    bool dPressed = false;

    bool previousLeftPressed = false;
    bool previousRightPressed = false;

    bool previousAPressed = false;
    bool previousDPressed = false;
};