#include "App.h"
#include "RenderContext.h"
#include "SimulationConfig.h"
#include "PBDSimulation.cuh"
#include "XPBDSimulation.cuh"

#include <iostream>
#include <exception>

int main() {
	try {
		SimulationConfig config;

		App app;
		if (!app.initialize(config.screenWidth, config.screenHeight, "PBD / XPBD Simulation")) {
			std::cerr << "App initialization failed." << std::endl;
			std::cin.get();
			return -1;
		}

		RenderContext renderContext;
		if (!renderContext.initialize()) {
			std::cerr << "RenderContext initialization failed." << std::endl;
			app.release();
			std::cin.get();
			return -1;
		}

		XPBDSimulation simulation(config);
		if (!simulation.initialize()) {
			std::cerr << "Simulation initialization failed." << std::endl;
			renderContext.release();
			app.release();
			std::cin.get();
			return -1;
		}
		double fpsMeasureStartTime = static_cast<double>(app.getCurrentTime());
		int renderedFrameCount = 0;
		bool leftMouseWasDown = false;

		while (!app.shouldClose()) {
			app.beginFrame();
			app.processInput();
			renderContext.updateMatrices(app.getCamera(), config.screenWidth, config.screenHeight);
			double mouseX = 0.0;
			double mouseY = 0.0;

			glfwGetCursorPos(app.getWindow(), &mouseX, &mouseY);
			bool leftMouseIsDown = glfwGetMouseButton(app.getWindow(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
			if (leftMouseIsDown && !leftMouseWasDown) {
				simulation.beginMouseDrag(mouseX, mouseY, renderContext.getProjection(), renderContext.getView(), config.screenWidth, config.screenHeight);
			}
			else if (leftMouseIsDown) {
				simulation.updateMouseDrag(mouseX, mouseY, renderContext.getProjection(), renderContext.getView(), config.screenWidth, config.screenHeight);
			}
			else if (!leftMouseIsDown && leftMouseWasDown) {
				simulation.endMouseDrag();
			}
			leftMouseWasDown = leftMouseIsDown;

			simulation.step(app.getCurrentTime());

			renderContext.updateMatrices(app.getCamera(), config.screenWidth, config.screenHeight);
			simulation.render(renderContext);

			app.endFrame();
			renderedFrameCount++;

			double currentTime = static_cast<double>(app.getCurrentTime());
			double elapsedTime = currentTime - fpsMeasureStartTime;

			if (elapsedTime >= 1.0) {
				double renderedFps = static_cast<double>(renderedFrameCount) / elapsedTime;

				std::cout<< "Actual rendered FPS: "<< renderedFps<< std::endl;

				fpsMeasureStartTime = currentTime;
				renderedFrameCount = 0;
			}
		}

		simulation.release();
		renderContext.release();
		app.release();

		return 0;
	}
	catch (const std::exception& e) {
		std::cerr << "Runtime error: " << e.what() << std::endl;
		std::cerr << "Press Enter to exit..." << std::endl;
		std::cin.get();
		return -1;
	}
}