#include "RenderContext.h"

#include <glm/gtc/matrix_transform.hpp>

bool RenderContext::initialize() {
	clothShader = new Shader("shader/PBD.vs", "shader/PBD.fs");
	floorShader = new Shader("shader/floor.vs", "shader/floor.fs");
	yarnShader = new Shader("shader/PBD.vs", "shader/PBD.fs");

	return true;
}

void RenderContext::updateMatrices(Camera& camera, unsigned int screenWidth, unsigned int screenHeight) {
	projection = glm::perspective(glm::radians(camera.Zoom), static_cast<float>(screenWidth) / static_cast<float>(screenHeight), 0.1f, 100.0f);
	view = camera.GetViewMatrix();
}

void RenderContext::release() {
	delete clothShader;
	delete floorShader;
	delete yarnShader;

	clothShader = nullptr;
	floorShader = nullptr;
	yarnShader = nullptr;
}

Shader& RenderContext::getClothShader() {
	return *clothShader;
}

Shader& RenderContext::getFloorShader() {
	return *floorShader;
}

Shader& RenderContext::getYarnShader() {
	return *yarnShader;
}

const glm::mat4& RenderContext::getProjection() const {
	return projection;
}

const glm::mat4& RenderContext::getView() const {
	return view;
}