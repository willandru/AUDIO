#include "Window.h"

#include <iostream>

// ============================================================
// CONSTRUCTOR
// ============================================================

Window::Window()
{
}


// ============================================================
// DESTRUCTOR
// ============================================================

Window::~Window()
{
    if (window != nullptr)
    {
        glfwDestroyWindow(window);
        window = nullptr;
    }

    glfwTerminate();
}


// ============================================================
// INICIALIZAR VENTANA
// ============================================================

bool Window::initialize()
{
    if (!glfwInit())
    {
        std::cerr << "Error inicializando GLFW.\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    glfwWindowHint(
        GLFW_OPENGL_PROFILE,
        GLFW_OPENGL_CORE_PROFILE
    );

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    if (monitor == nullptr)
    {
        std::cerr
            << "No se encontró el monitor principal.\n";

        glfwTerminate();
        return false;
    }

    const GLFWvidmode* videoMode =
        glfwGetVideoMode(monitor);

    if (videoMode == nullptr)
    {
        std::cerr
            << "No se pudo obtener el modo de video.\n";

        glfwTerminate();
        return false;
    }

    window = glfwCreateWindow(
        videoMode->width,
        videoMode->height,
        "AUDIO",
        monitor,
        nullptr
    );

    if (window == nullptr)
    {
        std::cerr
            << "Error creando ventana fullscreen.\n";

        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // ========================================================
    // GLAD
    // ========================================================

    if (!gladLoadGLLoader(
            reinterpret_cast<GLADloadproc>(
                glfwGetProcAddress)))
    {
        std::cerr
            << "Error inicializando GLAD.\n";

        glfwDestroyWindow(window);
        window = nullptr;

        glfwTerminate();

        return false;
    }

    // ========================================================
    // VIEWPORT
    // ========================================================

    glfwGetFramebufferSize(
        window,
        &width,
        &height
    );

    glViewport(
        0,
        0,
        width,
        height
    );

    std::cout
        << "OpenGL inicializado correctamente.\n"
        << "Version: "
        << glGetString(GL_VERSION)
        << '\n'
        << "Renderer: "
        << glGetString(GL_RENDERER)
        << '\n';

    return true;
}


// ============================================================
// EVENTOS
// ============================================================

void Window::pollEvents()
{
    glfwPollEvents();
}


// ============================================================
// ACTUALIZAR VIEWPORT
// ============================================================

void Window::updateViewport()
{
    if (window == nullptr)
    {
        return;
    }

    glfwGetFramebufferSize(
        window,
        &width,
        &height
    );

    glViewport(
        0,
        0,
        width,
        height
    );
}


// ============================================================
// PRESENTAR FRAME
// ============================================================

void Window::swapBuffers()
{
    if (window != nullptr)
    {
        glfwSwapBuffers(window);
    }
}


// ============================================================
// ESTADO DE LA VENTANA
// ============================================================

bool Window::shouldClose() const
{
    return window == nullptr ||
           glfwWindowShouldClose(window);
}


// ============================================================
// CERRAR VENTANA
// ============================================================

void Window::close()
{
    if (window != nullptr)
    {
        glfwSetWindowShouldClose(
            window,
            GLFW_TRUE
        );
    }
}


// ============================================================
// ACCESO A GLFW
// ============================================================

GLFWwindow* Window::getHandle() const
{
    return window;
}


// ============================================================
// DIMENSIONES
// ============================================================

int Window::getWidth() const
{
    return width;
}

int Window::getHeight() const
{
    return height;
}