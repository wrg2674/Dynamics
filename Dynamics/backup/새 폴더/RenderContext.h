#pragma once


#include <glm/glm.hpp>

#include <learnopengl/shader.h>
#include <learnopengl/camera.h>

class RenderContext {
public:
	RenderContext() = default;

	bool initialize();

	void updateMatrices(Camera& camera,unsigned int screenWidth,unsigned int screenHeight);

	void release();

	Shader& getClothShader();
	Shader& getFloorShader();
	Shader& getYarnShader();

	const glm::mat4& getProjection() const;
	const glm::mat4& getView() const;

private:
	Shader* clothShader = nullptr;
	Shader* floorShader = nullptr;
	Shader* yarnShader = nullptr;

	glm::mat4 projection = glm::mat4(1.0f);
	glm::mat4 view = glm::mat4(1.0f);
};