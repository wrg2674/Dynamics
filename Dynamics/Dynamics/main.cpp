#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <unordered_map>
#include <algorithm>
#include <glm/gtc/constants.hpp>

#include <iostream>
#include "teapot_loader.h"
#include "Vertex.h"
#include "StretchConstriant.h"
#include "BendingConstraint.h"
#include "PBD.h"

using namespace std;

struct EdgeKey {
	unsigned int a, b;
	bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};

struct EdgeKeyHash {
	size_t operator()(const EdgeKey& k) const {
		return (static_cast<size_t>(k.a) << 32) ^ static_cast<size_t>(k.b);
	}
};

struct EdgeInfo {
	unsigned int opp;
};

static inline EdgeKey makeKey(unsigned int u, unsigned int v) {
	if (u < v) return { u, v };
	return { v, u };
}



void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
void buildBendingConstraints_AllSharedEdges(vector<Vertex>& vertices, const vector<unsigned int>& indices, vector<Constraint*>& constraints, float K_BENDING, bool typeFlag);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float K_DAMPING = 0.3f;
const float TIMESTEP = 0.05f;
const int ROWS = 60;
const int COLS = 30;
const int ITERATION_COUNT = 50;
const float K_STRETCH = 1.0f;
const float K_BENDING = 0.5f;
const float GRAVITY = -9.8f;

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
			float px = 1 * j / (float)(COLS - 1) - 1 * (1 - j / (float)(COLS - 1));
			float py = -1 * i / (float)(ROWS - 1) + 1 * (1 - i / (float)(ROWS - 1));
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
	glBufferData(GL_ARRAY_BUFFER, cloth.size() * sizeof(Vertex), cloth.data(), GL_STATIC_DRAW);

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	// sizeof(indices)와 indices.size()*sizeof(unsigned int)는 다름
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
	glEnableVertexAttribArray(0);

	vector<glm::vec3> forces;
	glm::vec3 gravity = glm::vec3(0, GRAVITY, 0);
	forces.push_back(gravity);

	vector<Constraint*> constraints;
	float restX = 2.0f / (COLS - 1);
	float restY = 2.0f / (ROWS - 1);
	// 가로 stretch 제약
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS - 1; j++) {
			StretchConstraint* tmp = new StretchConstraint(2, K_STRETCH, true, 0);
			tmp->addVertex(&cloth.at(j + i * COLS));
			tmp->addVertex(&cloth.at(j + i * COLS + 1));
			tmp->d = tmp->calc();
			constraints.push_back(tmp);
		}
	}
	// 세로 stretch 제약
	for (int i = 0; i < ROWS - 1; i++) {
		for (int j = 0; j < COLS; j++) {
			StretchConstraint* tmp = new StretchConstraint(2, K_STRETCH, true, 0);
			tmp->addVertex(&cloth.at(j + i * COLS));
			tmp->addVertex(&cloth.at(j + (i + 1) * COLS));
			tmp->d = tmp->calc();
			constraints.push_back(tmp);
		}
	}
	// 대각선 stretch 제약
	for (int i = 0; i < ROWS - 1; i++) {
		for (int j = 0; j < COLS - 1; j++) {
			int p1 = i * COLS + j;
			int p2 = (i + 1) * COLS + j + 1;
			int p3 = i * COLS + j + 1;
			int p4 = (i + 1) * COLS + j;
			//if (p1 >= cloth.size() || p2 >= cloth.size() || p3 >= cloth.size() || p4 >= cloth.size()) {
			//	continue;
			//}
			StretchConstraint* tmp = new StretchConstraint(2, K_STRETCH, true, 0);
			tmp->addVertex(&cloth.at(p1));
			tmp->addVertex(&cloth.at(p2));
			tmp->d = tmp->calc();
			constraints.push_back(tmp);

			StretchConstraint* tmp2 = new StretchConstraint(2, K_STRETCH, true, 0);
			tmp2->addVertex(&cloth.at(p3));
			tmp2->addVertex(&cloth.at(p4));
			tmp2->d = tmp2->calc();
			constraints.push_back(tmp2);
		}
	}
	//buildBendingConstraints_AllSharedEdges(cloth, indices, constraints, K_BENDING, true);

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
	PBD pbd = PBD(cloth, forces, constraints, TIMESTEP, K_DAMPING, ITERATION_COUNT);
	while (!glfwWindowShouldClose(window)) {
		ios::sync_with_stdio(false);
		cin.tie(NULL);

		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glBufferSubData(GL_ARRAY_BUFFER, 0, cloth.size() * sizeof(Vertex), cloth.data());

		pbd.solve();

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

void buildBendingConstraints_AllSharedEdges(vector<Vertex>& vertices, const vector<unsigned int>& indices, vector<Constraint*>& constraints, float K_BENDING, bool typeFlag) {
	unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> edgeMap;
	edgeMap.reserve(indices.size());

	auto addEdge = [&](unsigned int u, unsigned int v, unsigned int opp) {
		EdgeKey key = makeKey(u, v);
		auto it = edgeMap.find(key);
		if (it == edgeMap.end()) {
			edgeMap.emplace(key, EdgeInfo{ opp });
			return;
		}
		unsigned int a = key.a;
		unsigned int b = key.b;
		unsigned int opp0 = it->second.opp;
		unsigned int opp1 = opp;

		BendingConstraint* bc = new BendingConstraint(4, K_BENDING, typeFlag, 0.0f);
		bc->addVertex(&vertices.at(a));
		bc->addVertex(&vertices.at(b));
		bc->addVertex(&vertices.at(opp0));
		bc->addVertex(&vertices.at(opp1));

		bc->phi = bc->calc();
		constraints.push_back(bc);
		edgeMap.erase(it);
	};
	for (size_t t = 0; t + 2 < indices.size(); t += 3) {
		unsigned int a = indices[t + 0];
		unsigned int b = indices[t + 1];
		unsigned int c = indices[t + 2];

		addEdge(a, b, c);
		addEdge(b, c, a);
		addEdge(c, a, b);
	}
}