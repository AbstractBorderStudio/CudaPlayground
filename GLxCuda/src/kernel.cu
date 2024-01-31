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
 *        VARIABLES
***************************************/

// global variables
static GLFWwindow* window;
static cudaError_t cudaStatus;

static double lastTime;
static int nFrames;

// data
typedef struct
{
    float x, y;
    float r, g, b;
} v_buffer;

static v_buffer vertices[3] =
{
    { -0.5f, -0.5f, 1.f, 0.f, 0.f },
    {  0.5f, -0.5f, 0.f, 1.f, 0.f },
    {   0.f,  .5f, 0.f, 0.f, 1.f }
};

/**************************************
 *        CUDA KERNELS
***************************************/

__global__
void UpdateVertexBufferInGPU(v_buffer* buffer)
{
    if (threadIdx.x == 0)
        buffer[threadIdx.x].x = buffer[threadIdx.x].x - 0.001f;
    else if (threadIdx.x == 1)
        buffer[threadIdx.x].x = buffer[threadIdx.x].x + 0.001f;
    else
        buffer[threadIdx.x].y = buffer[threadIdx.x].y + 0.001f;
}

/**************************************
 *        METHODS DEFINITIONS
***************************************/

static void error_callback(int error, const char* description);
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void Init(),
static void CreateWindow();
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void RenderLoop(GLFWwindow* _window);
void Draw();
void CleanUp(GLFWwindow* _window);
void FPSCounter();

/**************************************
 *              MAIN
***************************************/

static const char* vertex_shader_text =
"#version 110\n"
"uniform mat4 MVP;\n"
"attribute vec3 vCol;\n"
"attribute vec2 vPos;\n"
"varying vec3 color;\n"
"void main()\n"
"{\n"
"    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
"    // send the vertex color in output for the fragment shader to use\n"
"    color = vCol;\n"
"}\n";

static const char* fragment_shader_text =
"#version 110\n"
"varying vec3 color;\n"
"void main()\n"
"{\n"
"    // use the vertex color from the vertex buffer\n"
"    gl_FragColor = vec4(color, 1.0);\n"
"}\n";

static GLuint vertex_buffer, vertex_shader, fragment_shader, program;
static GLint mvp_location, vpos_location, vcol_location;

int main(void)
{
    // init system
    Init();
    // create window
    CreateWindow();

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
    glCompileShader(vertex_shader);
    
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
    glCompileShader(fragment_shader);
    
    program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    
    mvp_location = glGetUniformLocation(program, "MVP");
    vpos_location = glGetAttribLocation(program, "vPos");
    vcol_location = glGetAttribLocation(program, "vCol");
    
    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
        sizeof(vertices[0]), (void*)0);
    glEnableVertexAttribArray(vcol_location);
    glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE,
        sizeof(vertices[0]), (void*)(sizeof(float) * 2));

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
    //    // call 1 block 10 thread
    //    test << <1, 10 >> > ();
    //}
}

/**************************************
 *      APP INITIALIZATION
***************************************/

static void Init()
{
    // set glfw error callback
    glfwSetErrorCallback(error_callback);

    // init glfw
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
    // create window
    window = glfwCreateWindow(600, 600, "GLxCuda", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // set glfw window callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // create context
    glfwMakeContextCurrent(window);
    // load OpenGL functionalities
    gladLoadGL();
    // set
    glfwSwapInterval(1);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
    // Re-render the scene because the current frame was drawn for the old resolution
    Draw();
}

/**************************************
 *        RENDER LOOP
***************************************/

static v_buffer* cuda_vertices;
static double* m_time;

void RenderLoop(GLFWwindow* _window)
{
    // register cuda buffer
    cudaMalloc((void**)&cuda_vertices, 3 * sizeof(v_buffer));
    cudaMalloc((void**)&m_time, sizeof(double));

    while (!glfwWindowShouldClose(_window))
    {
        FPSCounter();

        // draw screen
        Draw();

        // poll events
        glfwPollEvents();
    }

    // free cuda buffer
    cudaFree(&cuda_vertices);
}

void Draw()
{
    float ratio;
    int width, height;
    mat4x4 m, p, mvp;

    glfwGetFramebufferSize(window, &width, &height);
    ratio = width / (float)height;

    // clear screen for the next frame
    glClear(GL_COLOR_BUFFER_BIT);
    // color screen
    glClearColor(0.1, 0.5, 1.0, 1.0);
    
    mat4x4_identity(m);
    mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
    mat4x4_mul(mvp, p, m);

    //*m_time = (double)glfwGetTime();
    //printf("%d\n", *m_time);

    // launch kernel
    cudaMemcpy((void*)cuda_vertices, vertices, 3 * sizeof(v_buffer), cudaMemcpyHostToDevice);
    UpdateVertexBufferInGPU << <1, 3 >> > (cuda_vertices);
    cudaMemcpy(vertices, (void*)cuda_vertices, 3 * sizeof(v_buffer), cudaMemcpyDeviceToHost);

    // update buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glUseProgram(program);
    glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*)mvp);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // swap buffer and pull events
    glfwSwapBuffers(window);
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