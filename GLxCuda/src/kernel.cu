/*
    MIT License

    Copyright (c) 2024 "Daniel Bologna & Erik Prifti"

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/


/**************************************
 *        INCLUDE LIBRARIES
***************************************/

// system includes
#include <stdlib.h>
#include <stdio.h>

// vector math includes
#include "linmath.h"

// opengl includes
#include <glad/glad.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

/**************************************
 *        CUDA KERNELS
***************************************/

__global__
void test()
{
    printf("Ciao\n");
}

/**************************************
 *        METHODS DEFINITIONS
***************************************/

static void error_callback(int error, const char* description);
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void Init(),
static void CreateWindow();
void RenderLoop(GLFWwindow* _window);
void CleanUp(GLFWwindow* _window);
void FPSCounter();

/**************************************
 *              MAIN
***************************************/

static GLFWwindow* window;
static cudaError_t cudaStatus;

static double lastTime;
static int nFrames;

int main(void)
{
    // init system
    Init();
    // create window
    CreateWindow();

    // rendering loop
    RenderLoop(window);

    // cleanup application
    CleanUp(window);
}

/**************************************
 *        ERORR HANDLER
***************************************/

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

/**************************************
 *        INPUT HANDLER
***************************************/

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    //else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    //{
    //    test << <10, 1 >> > ();
    //}
}

/**************************************
 *      APP INITIALIZATION
***************************************/

static void Init()
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // init fps counter
    lastTime = 0;
    nFrames = 0;
}

static void CreateWindow()
{
    window = glfwCreateWindow(480, 600, "GLxCuda", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetKeyCallback(window, key_callback);

    glfwMakeContextCurrent(window);
    gladLoadGL();
    glfwSwapInterval(1);
}

/**************************************
 *        RENDER LOOP
***************************************/

void RenderLoop(GLFWwindow* _window)
{
    while (!glfwWindowShouldClose(_window))
    {
        FPSCounter();

        // calculate screen ratio to scale content with the window
        //float ratio;
        int width, height;

        glfwGetFramebufferSize(_window, &width, &height);
        //ratio = width / (float)height;

        glViewport(0, 0, width, height);

        // clear screen for the next frame
        glClear(GL_COLOR_BUFFER_BIT);

        // color screen
        glClearColor(0.1, 0.5, 1.0, 1.0);

        // swap buffer and pull events
        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
}

/**************************************
 *        APP CLEANUP
***************************************/

void CleanUp(GLFWwindow* _window)
{
    glfwDestroyWindow(_window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}

/**************************************
 *        FPS COUNTER
***************************************/

void FPSCounter()
{
    double currentTime = glfwGetTime();
    static char strFPS[20] = { 0 };
    nFrames++;
    if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
        lastTime = currentTime;
        sprintf(strFPS, "GLxCuda - FPS: %d", nFrames);
        nFrames = 0;
        glfwSetWindowTitle(window, strFPS);
    }
}