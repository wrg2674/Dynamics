#include "App.h"
#include "RenderContext.h"
#include "SimulationConfig.h"
#include "PBDSimulation.cuh"

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

		PBDSimulation simulation(config);
		if (!simulation.initialize()) {
			std::cerr << "Simulation initialization failed." << std::endl;
			renderContext.release();
			app.release();
			std::cin.get();
			return -1;
		}

		while (!app.shouldClose()) {
			app.beginFrame();
			app.processInput();

			simulation.step(app.getCurrentTime());

			renderContext.updateMatrices(app.getCamera(), config.screenWidth, config.screenHeight);
			simulation.render(renderContext);

			app.endFrame();
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