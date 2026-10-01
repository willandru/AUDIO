#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Window
{
public:

    Window();
    ~Window();

    bool initialize();

    void pollEvents();
    void updateViewport();

    void swapBuffers();

    bool shouldClose() const;
    void close();

    GLFWwindow* getHandle() const;

    int getWidth() const;
    int getHeight() const;

private:

    GLFWwindow* window = nullptr;

    int width = 0;
    int height = 0;
};