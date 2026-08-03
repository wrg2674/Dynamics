#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <learnopengl/camera.h>

class App {
public:
	App();

	bool initialize(
		unsigned int width,
		unsigned int height,
		const char* title
	);

	bool shouldClose() const;

	void beginFrame();
	void endFrame();

	void processInput();

	void release();

	GLFWwindow* getWindow() const;
	Camera& getCamera();
	const Camera& getCamera() const;

	float getDeltaTime() const;
	float getCurrentTime() const;

	unsigned int getWidth() const;
	unsigned int getHeight() const;

private:
	static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
	static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

	void handleMouseMove(double xpos, double ypos);
	void handleScroll(double yoffset);

private:
	GLFWwindow* window = nullptr;

	unsigned int width = 800;
	unsigned int height = 600;

	Camera camera;

	float lastX = 400.0f;
	float lastY = 300.0f;
	bool firstMouse = true;

	float deltaTime = 0.0f;
	float lastFrame = 0.0f;
};