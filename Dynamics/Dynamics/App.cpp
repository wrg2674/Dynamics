#include "App.h"

#include <iostream>

App::App()
	: camera(glm::vec3(0.0f, 0.0f, 3.0f)) {
}

bool App::initialize(unsigned int width_,unsigned int height_,const char* title) {
	width = width_;
	height = height_;

	lastX = width * 0.5f;
	lastY = height * 0.5f;

	if (!glfwInit()) {
		std::cout << "Failed to initialize GLFW" << std::endl;
		return false;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (window == nullptr) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return false;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	glfwSetWindowUserPointer(window, this);

	glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetScrollCallback(window, scrollCallback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetWindowPos(window, -1000, 50);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return false;
	}

	glEnable(GL_DEPTH_TEST);
	glPointSize(2.0f);

	return true;
}

bool App::shouldClose() const {
	return glfwWindowShouldClose(window);
}

void App::beginFrame() {
	float currentFrame = static_cast<float>(glfwGetTime());

	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void App::endFrame() {
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void App::processInput() {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera.ProcessKeyboard(FORWARD, deltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera.ProcessKeyboard(LEFT, deltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera.ProcessKeyboard(RIGHT, deltaTime);
	}
}

void App::release() {
	glfwTerminate();
	window = nullptr;
}

GLFWwindow* App::getWindow() const {
	return window;
}

Camera& App::getCamera() {
	return camera;
}

const Camera& App::getCamera() const {
	return camera;
}

float App::getDeltaTime() const {
	return deltaTime;
}

float App::getCurrentTime() const {
	return static_cast<float>(glfwGetTime());
}

unsigned int App::getWidth() const {
	return width;
}

unsigned int App::getHeight() const {
	return height;
}

void App::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}

void App::mouseCallback(GLFWwindow* window, double xpos, double ypos) {
	App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
	if (app == nullptr) {
		return;
	}

	app->handleMouseMove(xpos, ypos);
}

void App::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	App* app = static_cast<App*>(glfwGetWindowUserPointer(window));
	if (app == nullptr) {
		return;
	}

	app->handleScroll(yoffset);
}

void App::handleMouseMove(double xpos_, double ypos_) {
	float xpos = static_cast<float>(xpos_);
	float ypos = static_cast<float>(ypos_);

	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

void App::handleScroll(double yoffset) {
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}