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


#include <app.h>

/**************************************
 *        VARIABLES
***************************************/
static GLFWwindow* window;
static cudaError_t cudaStatus;

static double m_time, delta_last, m_delta;

static double fps_lastTime;
static int nFrames;

static v_buffer vertices[3] =
{
    { -0.5f, -0.5f, 1.f, 0.f, 0.f },
    {  0.5f, -0.5f, 0.f, 1.f, 0.f },
    {   0.f,  .5f, 0.f, 0.f, 1.f }
};

static GLuint vertex_buffer, vertex_shader, fragment_shader, program;
static GLint mvp_location, vpos_location, vcol_location;

static v_buffer* cuda_vertices;

static bool enabled = false;

/**************************************
 *        CUDA KERNELS
***************************************/

__global__
void UpdateVertexBufferInGPU(v_buffer* buffer, double time, double delta)
{
    if (threadIdx.x == 0)
        buffer[threadIdx.x].x = sin(time) / 2.f - .5f;
    else if (threadIdx.x == 1)
        buffer[threadIdx.x].x = -sin(time) / 2.f + .5f;
    else
        buffer[threadIdx.x].y = -sin(time) / 2.f + .5f;
}

/**************************************
 *              MAIN
***************************************/

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
    else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        enabled = !enabled;
    }
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
    fps_lastTime = 0;
    nFrames = 0;
    m_time = delta_last = m_delta = 0.f;
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
    glfwSwapInterval(0);
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

void RenderLoop(GLFWwindow* _window)
{
    // register cuda buffer
    cudaMalloc((void**)&cuda_vertices, 3 * sizeof(v_buffer));

    while (!glfwWindowShouldClose(_window))
    {
        ComputeDeltaTime();
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

    glUseProgram(program);
    glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*)mvp);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // launch kernel
    if (enabled)
    {
        cudaMemcpy((void*)cuda_vertices, vertices, 3 * sizeof(v_buffer), cudaMemcpyHostToDevice);
        UpdateVertexBufferInGPU << <1, 3 >> > (cuda_vertices, m_time, m_delta);
        cudaMemcpy(vertices, (void*)cuda_vertices, 3 * sizeof(v_buffer), cudaMemcpyDeviceToHost);
    }

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
    // destroy window
    glfwDestroyWindow(_window);

    // close app
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

/**************************************
 *        DELTA TIME
***************************************/

void ComputeDeltaTime()
{
    m_time = glfwGetTime();
    m_delta = m_time - delta_last;
    delta_last = m_time;
}

/**************************************
 *        FPS COUNTER
***************************************/

void FPSCounter()
{
    // compute frame per second (1000.0 ms)
    static char strFPS[20] = { 0 };
    nFrames++;
    if (m_time - fps_lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
        fps_lastTime = m_time;
        sprintf(strFPS, "GLxCuda - FPS: %d - Delta: %f", nFrames, m_delta);
        nFrames = 0;
        // update window title
        glfwSetWindowTitle(window, strFPS);
    }
}