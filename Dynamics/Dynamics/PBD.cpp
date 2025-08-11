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
void sumExtForce(vector<Vertex>& vertices, vector<glm::vec3>& forces, vector<glm::vec3>& result);
void dampVelocities(vector<Vertex>& vertices);
void estimateP(vector<Vertex>& vertices, float tstep);
void GSiteration(vector<Constraint*>& constraints, float tstep);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float K_DAMPING = 0.3f;
const float TIMESTEP = 0.05f;
const int ROWS = 60;
const int COLS = 30;
const int ITERATION_COUNT = 30;
const float K = 1.0f;
const float GRAVITY = -9.8f;

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
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			float px = 1 * j / (float)(COLS-1) - 1 * (1 - j/ (float)(COLS-1));
			float py = -1 * i / (float)(ROWS-1) + 1 * (1 - i/ (float)(ROWS-1));
			float pz = 0;
			Vertex tmp = Vertex(px, py, pz, 0.0f, 0.0f, 0.0f, 1.0f);
			cloth.push_back(tmp);
		}
	}

	std::vector<unsigned int> indices;
	for (int i = 0; i < ROWS - 1; ++i) {
		for (int j = 0; j < COLS - 1; ++j) {
			int i0 = i * COLS + j;           // 좌상단
			int i1 = i * COLS + (j + 1);     // 우상단
			int i2 = (i + 1) * COLS + j;     // 좌하단
			int i3 = (i + 1) * COLS + (j + 1); // 우하단

			// 삼각형 1: (i0, i1, i2)
			indices.push_back(i0);
			indices.push_back(i1);
			indices.push_back(i2);

			// 삼각형 2: (i1, i3, i2)
			indices.push_back(i1);
			indices.push_back(i3);
			indices.push_back(i2);
		}
	}

	GLuint VAO, VBO, EBO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, cloth.size()* sizeof(Vertex), cloth.data(), GL_STATIC_DRAW);

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	// sizeof(indices)와 indices.size()*sizeof(unsigned int)는 다름
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, x));
	glEnableVertexAttribArray(0);

	vector<glm::vec3> forces;
	glm::vec3 gravity = glm::vec3(0, GRAVITY, 0);
	forces.push_back(gravity);

	vector<Constraint*> distanceConstraints;
	float restX = 2.0f / (COLS - 1);
	float restY = 2.0f / (ROWS - 1);
	// 양옆간의 거리제약
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS-1; j++) {
			DistanceConstraint* tmp = new DistanceConstraint(2, K, true, restX);
			tmp->addVertex(&cloth.at(j+i*COLS));
			tmp->addVertex(&cloth.at(j+i*COLS+1));
			distanceConstraints.push_back(tmp);
		}
	}
	// 위아래간의 거리제약
	for (int i = 0; i < ROWS-1; i++) {
		for (int j = 0; j < COLS; j++) {
			DistanceConstraint* tmp = new DistanceConstraint(2, K, true, restY);
			tmp->addVertex(&cloth.at(j + i * COLS));
			tmp->addVertex(&cloth.at(j + (i+1) * COLS ));
			distanceConstraints.push_back(tmp);
		}
	}
	glPointSize(2.0f);
	int count = 0;
	int stopper = 0;
	//cin >> stopper;

	cloth.at(0).pinned = true;
	cloth.at(COLS - 1).pinned = true;
	
	for (int i = 0; i < cloth.size(); i++) {
		if (cloth.at(i).pinned == true) {
			cloth.at(i).m = 100000;
		}
	}

	while (!glfwWindowShouldClose(window)) {
		ios::sync_with_stdio(false);
		cin.tie(NULL);

		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		applyForce(cloth, forces, TIMESTEP);
		dampVelocities(cloth);
		estimateP(cloth, TIMESTEP);
		// generateCollisionConstraint();
		GSiteration(distanceConstraints, TIMESTEP);
		updateVertices(cloth, TIMESTEP);
		// velocityUpdate();

		glBufferSubData(GL_ARRAY_BUFFER, 0, cloth.size() * sizeof(Vertex), cloth.data());
		
		ourShader.use();
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		ourShader.setMat4("projection", projection);

		glm::mat4 view = camera.GetViewMatrix();
		ourShader.setMat4("view", view);

		glm::mat4 model = glm::mat4(1.0f);
		//model = glm::scale(model, glm::vec3(10, 10, 1));
		ourShader.setMat4("model", model);

		glBindVertexArray(VAO);
		//glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
		glDrawElements(GL_POINTS, indices.size(), GL_UNSIGNED_INT, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
		count++;
	}
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	glfwTerminate();
	return 0;
}

void sumExtForce(vector<Vertex>& vertices, vector<glm::vec3>& forces, vector<glm::vec3>& result) {
	for (int k = 0; k < vertices.size(); k++) {
		for (int i = 0; i < forces.size(); i++) {
			for (int j = 0; j < 3; j++) {
				result[k][j] += forces.at(i)[j] * vertices.at(k).m;
			}
		}
	}
	
}
void applyForce(vector<Vertex>& vertices, vector<glm::vec3>& forces, float tstep) {
	vector<glm::vec3> vertexForce = vector<glm::vec3>();
	for (int i = 0; i < vertices.size(); i++) {
		vertexForce.push_back({ 0,0,0 });
	}
	sumExtForce(vertices, forces, vertexForce);
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			cur.v[j] = cur.v[j] + tstep * (1.0 / cur.m) * vertexForce[i][j];
			if (cur.pinned) {
				cur.v[j] = 0;
			}
		}
	}
}
void dampVelocities(vector<Vertex>& vertices) {
	glm::vec3 xcm = { 0,0,0 };
	glm::vec3 vcm = { 0,0,0 };
	glm::vec3 sumXm = { 0,0,0 };
	glm::vec3 sumVm = { 0,0,0 };
	float sumM = 0;
	glm::vec3 L = { 0,0,0 }, w = {0,0,0};
	glm::mat3 I=glm::mat3(1.0f);
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
		cur.p = cur.x + cur.v * tstep;
		if (cur.pinned) {
			cur.p = cur.x;
		}
	}
}
CollisionDetection CCD(vector<Vertex>& vertices) {
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		glm::vec3 ray = { 0,0,0 };
		for (int j = 0; j < 3; j++) {
			ray[j] = cur.p[j] - cur.x[j];
		}

	}
	return FAIL;
}
void generateCollisionConstraint(vector<Vertex>& vertices) {

}

void GSiteration(vector<Constraint*>& constraints, float tstep) {
	// GS 스타일의 즉시 업데이트는 제약사항 단위의 것을 의미하는 것이지 
	// 한 제약사항 내에서 각 정점마다 즉시 업데이트를 하면 안됨.
	for (int count = 0; count < ITERATION_COUNT; count++) {
		for (int i = 0; i < constraints.size(); i++) {
			constraints.at(i)->projectionFunction(tstep, ITERATION_COUNT);
		}
	}

}
void updateVertices(vector<Vertex>& vertices, float tstep) {
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			cur.v[j] = (cur.p[j] - cur.x[j]) / tstep;
			if (cur.pinned) {
				cur.v[j] = 0.0f;
			}
		}
		for (int j = 0; j < 3; j++) {
			cur.x[j] = cur.p[j];
			if (cur.pinned) {
				cur.x[j] = cur.x[j];
			}
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