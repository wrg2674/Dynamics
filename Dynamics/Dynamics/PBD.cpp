#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include "teapot_loader.h"
#include "Vertex.h"
#include "distanceConstriant.h"

using namespace std;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

void velocityUpdate(vector<Vertex>& vertices);
void updateVertices(vector<Vertex>& vertices, float tstep);
void applyForce(vector<Vertex>& vertices, vector<glm::vec3>& forces, float tstep);
void sumExtForce(vector<glm::vec3>& forces, glm::vec3 result);
void dampVelocities(vector<Vertex>& vertices);
void estimateP(vector<Vertex>& vertices, float tstep);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float K_DAMPING = 0.3;

enum CollisionDetection { TRUE, FALSE, FAIL };

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
float lastFrame = 0.0f;

glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

int main() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "wrg", NULL, NULL);
	if (window == NULL) {
		cout << "Failed to create GLFW window\n";
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		cout << "Failed to initiallize GLAD\n";
		return -1;
	}
	glEnable(GL_DEPTH_TEST);

	Shader ourShader("shader/PBD.vs", "shader/PBD.fs");



	vector<Vertex> cloth;
	//for (int i = 0; i < 50; i++) {
	//	for (int j = 0; j < 20; j++) {
	//		float px = -1 * j / 20.0f + 1 * (1 - j / 20.0f);
	//		float py = -1 * i / 50.0f + 1 * (1 - i / 50.0f);
	//		float pz = 0;
	//		Vertex tmp = Vertex(px, py, pz, 0.0f, 0.0f, 0.0f, 1.0f);
	//		cloth.push_back(tmp);
	//	}
	//}
	
	for (int i = 0; i < 60; i++) {
		for (int j = 0; j < 30; j++) {
			float x1 = -1 * j / 30.0f + 1 * (1 - j) / 30.0f;
			float y1 = -1 * i / 60.0f + 1 * (1 - i) / 60.0f;

			float x2 = -1 * (j + 1) / 30.0f + 1 * (1 - (j + 1)) / 30.0f;
			float y2 = -1 * i / 60.0f + 1 * (1 - i) / 60.0f;

			float x3 = -1 * j / 30.0f + 1 * (1 - j) / 30.0f;
			float y3 = -1 * (i+1) / 60.0f + 1 * (1 - (i+1)) / 60.0f;

			Vertex tmp1 = Vertex(x1, y1, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			Vertex tmp2 = Vertex(x2, y2, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			Vertex tmp3 = Vertex(x3, y3, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			cloth.push_back(tmp1);
			cloth.push_back(tmp2);
			cloth.push_back(tmp3);
		}
	}
	for (int i = 60; i > 0; i--) {
		for (int j = 30; j > 0; j--) {
			float x1 = -1 * j / 30.0f + 1 * (1 - j) / 30.0f;
			float y1 = -1 * i / 60.0f + 1 * (1 - i) / 60.0f;

			float x2 = -1 * (j - 1) / 30.0f + 1 * (1 - (j - 1)) / 30.0f;
			float y2 = -1 * i / 60.0f + 1 * (1 - i) / 60.0f;

			float x3 = -1 * j / 30.0f + 1 * (1 - j) / 30.0f;
			float y3 = -1 * (i - 1) / 60.0f + 1 * (1 - (i - 1)) / 60.0f;

			Vertex tmp1 = Vertex(x1, y1, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			Vertex tmp2 = Vertex(x2, y2, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			Vertex tmp3 = Vertex(x3, y3, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
			cloth.push_back(tmp1);
			cloth.push_back(tmp2);
			cloth.push_back(tmp3);
		}
	}

	GLuint VAO, VBO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, cloth.size()* sizeof(Vertex), cloth.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, x));
	glEnableVertexAttribArray(0);

	DistanceConstraint distanceConstraint(2, 0.5, true, 1.0);
	glm::vec3 gravity = glm::vec3(0, -9.8, 0);

	while (!glfwWindowShouldClose(window)) {
		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		ourShader.use();
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		ourShader.setMat4("projection", projection);

		glm::mat4 view = camera.GetViewMatrix();
		ourShader.setMat4("view", view);

		glm::mat4 model = glm::mat4(1.0f);
		model = glm::scale(model, glm::vec3(0.8, 0.8, 1));
		ourShader.setMat4("model", model);

		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, cloth.size());
		//glDrawArrays(GL_POINTS, 0, cloth.size());

		glfwSwapBuffers(window);
		glfwPollEvents();

	}
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glfwTerminate();
	return 0;
}

void sumExtForce(vector<glm::vec3>& forces, glm::vec3 result) {
	for (int i = 0; i < forces.size(); i++) {
		for (int j = 0; j < 3; j++) {
			result[j] += forces.at(i)[j];
		}
	}
}
void applyForce(vector<Vertex>& vertices, vector<glm::vec3>& forces, float tstep) {
	glm::vec3 force = { 0,0,0 };
	sumExtForce(forces, force);
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			cur.v[j] = cur.v[j] + tstep * (1.0 / cur.m) * force[j];
		}
	}
}
void dampVelocities(vector<Vertex>& vertices) {
	glm::vec3 xcm = { 0,0,0 };
	glm::vec3 vcm = { 0,0,0 };
	glm::vec3 sumXm = { 0,0,0 };
	glm::vec3 sumVm = { 0,0,0 };
	float sumM = 0;
	glm::vec3 L, w;
	glm::mat3 I;
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			sumXm[j] += cur.x[j] * cur.m;
			sumVm[j] += cur.v[j] * cur.m;
			sumM += cur.m;
		}
	}
	for (int j = 0; j < 3; j++) {
		xcm[j] = sumXm[j] / sumM;
		vcm[j] = sumVm[j] / sumM;
	}
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		glm::vec3 r = { 0,0,0 };
		for (int j = 0; j < 3; j++) {
			r[j] = cur.x[j] - xcm[j];
		}
		L += glm::cross(r, cur.m * cur.v);
		glm::mat3 skew = { {0, r[2], -r[1]}, {-r[2], 0, r[0]}, {r[1], -r[0], 0} };
		I += skew * glm::transpose(skew) * cur.m;
	}
	w = glm::inverse(I) * L;
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		glm::vec3 deltaV = vcm + glm::cross(w, cur.x - xcm) - cur.v;
		cur.v = cur.v + K_DAMPING * deltaV;
	}
}

void estimateP(vector<Vertex>& vertices, float tstep) {
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			cur.p[j] = cur.x[j]+cur.v[j]*tstep;
		}
	}
}
CollisionDetection CCD(vector<Vertex>& vertices) {
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		float ray[3] = { 0,0,0 };
		for (int j = 0; j < 3; j++) {
			ray[j] = cur.p[j] - cur.x[j];
		}

	}
	return FAIL;
}
void generateCollisionConstraint(vector<Vertex>& vertices) {

}
void updateVertices(vector<Vertex>& vertices, float tstep) {
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			cur.v[j] = (cur.p[j] - cur.x[j]) / tstep;
		}
		for (int j = 0; j < 3; j++) {
			cur.x[j] = cur.p[j];
		}
	}
}
void velocityUpdate(vector<Vertex>& vertices) {

}

void processInput(GLFWwindow* window) {
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
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}
void mouse_callback(GLFWwindow* window, double xposln, double yposln) {
	float xpos = static_cast<float>(xposln);
	float ypos = static_cast<float>(yposln);

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

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	camera.ProceessMouseScroll(static_cast<float>(yoffset));
}