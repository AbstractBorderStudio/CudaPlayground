#ifndef _H_APP_H
#define _H_APP_H
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
 *        METHODS DEFINITIONS
***************************************/

static void error_callback(int error, const char* description);
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void Init(void),
static void CreateWindow(void);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void RenderLoop(GLFWwindow* _window);
void Draw(void);
void CleanUp(GLFWwindow* _window);
void ComputeDeltaTime(void);
void FPSCounter(void);

/**************************************
 *             DATA
***************************************/
typedef struct
{
    float x, y;
    float r, g, b;
} v_buffer;

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
#endif // !_H_APP_H