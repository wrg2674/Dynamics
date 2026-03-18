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
#include <cuda_gl_interop.h>

#include <iostream>
#include "teapot_loader.h"
#include "Vertex.h"
#include "Constraint.h"
#include "Damping.h"
#include "PBDSolver.cuh"

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
static void checkCuda(cudaError_t err, const char* msg);
static void buildClothPositions(vector<glm::vec3>& h_pos, vector<glm::vec3>& h_v, vector<glm::vec3>& h_p, vector<glm::vec3>& h_dp, vector<float>& h_invM);
static void buildTriangleIndices(vector<unsigned int>& indices);
static void buildStretchConstraints(ConstraintHost& cons, const vector<glm::vec3>& h_pos);
static void buildBendingConstraints_AllSharedEdges(vector<unsigned int>& indices, vector<glm::vec3>& h_pos, ConstraintHost& cons);
static void buildPerVertexConstraintLists(const ConstraintHost& cons, vector<unsigned int>& constraintsArray, vector<int>& constraintOffset, int vertexCount);
static void buildStretchColorBatches(ConstraintHost& cons);
static void buildBendingColorBatches(ConstraintHost& cons);
static void uploadColorBatch(const ColorBatchHost& host, ColorBatchDevice& device);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
const float K_DAMPING = 0.3f;
const float TIMESTEP = 0.05f;
const int ROWS = 30;
const int COLS = 30;
const int ITERATION_COUNT = 30;
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

	checkCuda(cudaGLSetGLDevice(0), "cudaGLSetGLDevice failed");
	
	glEnable(GL_DEPTH_TEST);
	glPointSize(2.0f);

	Shader ourShader("shader/PBD.vs", "shader/PBD.fs");

	const int vertexCount = ROWS * COLS;
	vector<glm::vec3> h_pos;
	vector<glm::vec3> h_v;
	vector<glm::vec3> h_p;
	vector<glm::vec3> h_dp;
	vector<float> h_invM;

	buildClothPositions(h_pos, h_v, h_p, h_dp, h_invM);

	ConstraintHost h_cons;
	buildStretchConstraints(h_cons, h_pos);

	vector<unsigned int> triangleIndices;
	buildTriangleIndices(triangleIndices);

	buildBendingConstraints_AllSharedEdges(triangleIndices, h_pos, h_cons);

	buildStretchColorBatches(h_cons);
	buildBendingColorBatches(h_cons);
	
	vector<unsigned int> h_constraintsArray;
	vector<int> h_constraintOffset;
	buildPerVertexConstraintLists(h_cons, h_constraintsArray, h_constraintOffset, vertexCount);

	GLuint VAO, VBO, EBO;
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &EBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	// sizeof(indices)와 indices.size()*sizeof(unsigned int)는 다름
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangleIndices.size() * sizeof(unsigned int), triangleIndices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	cudaGraphicsResource* cudaVBO = nullptr;
	checkCuda(cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsMapFlagsNone), "cudaGraphicsGLRegisterBuffer failed");

	checkCuda(cudaGraphicsMapResources(1, &cudaVBO), "cudaGraphicsMapResources(init) failed");
	glm::vec3* d_vboPos = nullptr;
	size_t mappedSize = 0;
	checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_vboPos, &mappedSize, cudaVBO), "cudaGraphicsResourceGetMappedPointer(init) failed");
	checkCuda(cudaMemcpy(d_vboPos, h_pos.data(), sizeof(float3) * vertexCount, cudaMemcpyHostToDevice), "Initial copy to VBO failed");
	checkCuda(cudaGraphicsUnmapResources(1, &cudaVBO), "cudaGraphicsUnmapResources(init) failed");

	VertexDevice d_ver{};
	d_ver.pos = nullptr;
	d_ver.N = vertexCount;
	d_ver.constraintNum = (int)h_constraintsArray.size();

	checkCuda(cudaMalloc(&d_ver.v, sizeof(float3) * vertexCount), "cudaMalloc d_ver.v failed");
	checkCuda(cudaMalloc(&d_ver.p, sizeof(float3) * vertexCount), "cudaMalloc d_ver.p failed");
	checkCuda(cudaMalloc(&d_ver.dp, sizeof(float3) * vertexCount), "cudaMalloc d_ver.dp failed");
	checkCuda(cudaMalloc(&d_ver.invM, sizeof(float) * vertexCount), "cudaMalloc d_Ver.invM failed");
	checkCuda(cudaMalloc(&d_ver.constraintsArray, sizeof(unsigned int) * h_constraintsArray.size()), "cudaMalloc d_ver.constraintsArray failed");
	checkCuda(cudaMalloc(&d_ver.constraintOffset, sizeof(int) * (vertexCount + 1)), "cudaMalloc d_ver.constraintOffset failed");

	checkCuda(cudaMemcpy(d_ver.v, h_v.data(), sizeof(float3) * vertexCount, cudaMemcpyHostToDevice), "cudaMemcpy d_ver.v failed");
	checkCuda(cudaMemcpy(d_ver.p, h_p.data(), sizeof(float3) * vertexCount, cudaMemcpyHostToDevice), "cudaMemcpy d_ver.p failed");
	checkCuda(cudaMemcpy(d_ver.dp, h_dp.data(), sizeof(float3) * vertexCount, cudaMemcpyHostToDevice), "cudaMemcpy d_ver.dp failed");
	checkCuda(cudaMemcpy(d_ver.invM, h_invM.data(), sizeof(float) * vertexCount, cudaMemcpyHostToDevice), "cudaMemcpy d_ver.invM failed");
	checkCuda(cudaMemcpy(d_ver.constraintsArray, h_constraintsArray.data(), sizeof(unsigned int) * h_constraintsArray.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_ver.constraintsArray failed");
	checkCuda(cudaMemcpy(d_ver.constraintOffset, h_constraintOffset.data(), sizeof(int) * (vertexCount + 1), cudaMemcpyHostToDevice), "cudaMemcpy d_ver.constraintOffset failed");

	ConstraintDevice d_cons{};

	d_cons.stretch.color.constraintIds = nullptr;
	d_cons.stretch.color.colorOffset = nullptr;
	d_cons.stretch.color.colorCount = 0;

	d_cons.bending.color.constraintIds = nullptr;
	d_cons.bending.color.colorOffset = nullptr;
	d_cons.bending.color.colorCount = 0;

	d_cons.stretch.n = h_cons.stretch.ver.size();
	d_cons.bending.n = h_cons.bending.ver.size();
	
	checkCuda(cudaMalloc(&d_cons.stretch.ver, sizeof(int2) * d_cons.stretch.n), "cudaMalloc stretch.ver failed");
	checkCuda(cudaMalloc(&d_cons.stretch.k, sizeof(float) * d_cons.stretch.n), "cudaMalloc stretch.k failed");
	checkCuda(cudaMalloc(&d_cons.stretch.l0, sizeof(float) * d_cons.stretch.n), "cudaMalloc stretch.l0 failed");
	if (d_cons.stretch.n > 0) {
		checkCuda(cudaMemcpy(d_cons.stretch.ver, h_cons.stretch.ver.data(), sizeof(int2) * d_cons.stretch.n, cudaMemcpyHostToDevice), "cudaMemcpy stretch.ver failed");
		checkCuda(cudaMemcpy(d_cons.stretch.k, h_cons.stretch.k.data(), sizeof(float) * d_cons.stretch.n, cudaMemcpyHostToDevice), "cudaMemcpy stretch.k failed");
		checkCuda(cudaMemcpy(d_cons.stretch.l0, h_cons.stretch.l0.data(), sizeof(float) * d_cons.stretch.n, cudaMemcpyHostToDevice), "cudaMemcpy stretch.l0 failed");

	}
	checkCuda(cudaMalloc(&d_cons.bending.ver, sizeof(int4) * d_cons.bending.n), "cudaMalloc bending.ver failed");
	checkCuda(cudaMalloc(&d_cons.bending.k, sizeof(float) * d_cons.bending.n), "cudaMalloc bending.k failed");
	checkCuda(cudaMalloc(&d_cons.bending.phi0, sizeof(float) * d_cons.bending.n), "cudaMalloc bending.phi0 failed");

	if (d_cons.bending.n > 0) {
		checkCuda(cudaMemcpy(d_cons.bending.ver, h_cons.bending.ver.data(), sizeof(int4)* d_cons.bending.n, cudaMemcpyHostToDevice), "cudaMemcpy bending.ver failed");
		checkCuda(cudaMemcpy(d_cons.bending.k, h_cons.bending.k.data(), sizeof(float) * d_cons.bending.n, cudaMemcpyHostToDevice), "cudaMemcpy bending.k failed");
		checkCuda(cudaMemcpy(d_cons.bending.phi0, h_cons.bending.phi0.data(), sizeof(float) * d_cons.bending.n, cudaMemcpyHostToDevice), "cudaMemcpy bending.phi0 failed");
	}

	uploadColorBatch(h_cons.stretch.color, d_cons.stretch.color);
	uploadColorBatch(h_cons.bending.color, d_cons.bending.color);

	DampingDevice d_damp{};
	checkCuda(cudaMalloc(&d_damp.poscm, sizeof(float3)), "cudaMalloc damp.poscm failed");
	checkCuda(cudaMalloc(&d_damp.vcm, sizeof(float3)), "cudaMalloc damp.vcm failed");
	checkCuda(cudaMalloc(&d_damp.omega, sizeof(float3)), "cudaMalloc damp.omega failed");

	vector<glm::vec3> h_forces;
	glm::vec3 gravity = glm::vec3(0.0f, GRAVITY, 0.0f);
	h_forces.push_back(gravity);

	float3* d_forces = nullptr;
	checkCuda(cudaMalloc(&d_forces, sizeof(float3) * h_forces.size()), "cudaMalloc d_forces failed");
	checkCuda(cudaMemcpy(d_forces, h_forces.data(), sizeof(float3) * h_forces.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_forces failed");

	while (!glfwWindowShouldClose(window)) {
		
		float currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		processInput(window);
		checkCuda(cudaGraphicsMapResources(1, &cudaVBO), "cudaGraphicsMapResources(frame) failed");

		float3* d_vboPos = nullptr;
		size_t mappedSize = 0;
		checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_vboPos, &mappedSize, cudaVBO), "cudaGraphicsResourceGetMappedPointer(frame) failed");
		
		d_ver.pos = d_vboPos;
		solve(d_ver, d_cons, d_damp, d_forces, K_DAMPING, TIMESTEP, ITERATION_COUNT, (int)h_forces.size(), vertexCount, h_cons.stretch.color.colorOffset, h_cons.bending.color.colorOffset);

		checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after solve failed");
		checkCuda(cudaGraphicsUnmapResources(1, &cudaVBO), "cudaGraphicsUnmapResources(frame) failed");


		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		ourShader.use();
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		ourShader.setMat4("projection", projection);

		glm::mat4 view = camera.GetViewMatrix();
		ourShader.setMat4("view", view);

		glm::mat4 model = glm::mat4(1.0f);
		//model = glm::scale(model, glm::vec3(10, 10, 1));
		ourShader.setMat4("model", model);

		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS, 0, vertexCount);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	cudaGraphicsUnregisterResource(cudaVBO);

	cudaFree(d_forces);

	cudaFree(d_damp.poscm);
	cudaFree(d_damp.vcm);
	cudaFree(d_damp.omega);

	cudaFree(d_cons.stretch.ver);
	cudaFree(d_cons.stretch.k);
	cudaFree(d_cons.stretch.l0);

	cudaFree(d_cons.bending.ver);
	cudaFree(d_cons.bending.k);
	cudaFree(d_cons.bending.phi0);

	cudaFree(d_cons.stretch.color.constraintIds);
	cudaFree(d_cons.stretch.color.colorOffset);

	cudaFree(d_cons.bending.color.constraintIds);
	cudaFree(d_cons.bending.color.colorOffset);

	cudaFree(d_ver.v);
	cudaFree(d_ver.p);
	cudaFree(d_ver.dp);
	cudaFree(d_ver.invM);
	cudaFree(d_ver.constraintsArray);
	cudaFree(d_ver.constraintOffset);

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
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}
static void checkCuda(cudaError_t err, const char* msg) {
	if (err != cudaSuccess) {
		cerr << msg << " : " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}
static void buildClothPositions(vector<glm::vec3>& h_pos, vector<glm::vec3>& h_v, vector<glm::vec3>& h_p, vector<glm::vec3>& h_dp, vector<float>& h_invM) {
	const int N = ROWS * COLS;

	h_pos.resize(N);
	h_v.resize(N);
	h_p.resize(N);
	h_dp.resize(N);
	h_invM.resize(N);

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
			int idx = i * COLS + j;
			float px = j / (float)(COLS - 1) - (1.0f - j / (float)(COLS - 1));
			float py = -i / (float)(ROWS - 1) + (1.0f - i / (float)(ROWS - 1));
			float pz = 0.0f;

			h_pos[idx] = glm::vec3(px, py, pz);
			h_v[idx] = glm::vec3(0);
			h_p[idx] = h_pos[idx];
			h_dp[idx] = glm::vec3(0);
			h_invM[idx] = 1.0f;
		}
	}
	h_invM[0] = 0.0f;
	h_invM[COLS - 1] = 0.0f;
}
static void buildTriangleIndices(vector<unsigned int>& indices) {
	indices.clear();
	for (int i = 0; i < ROWS - 1; i++) {
		for (int j = 0; j < COLS - 1; j++) {
			int i0 = i * COLS + j;
			int i1 = i * COLS + j + 1;
			int i2 = (i + 1) * COLS + j;
			int i3 = (i + 1) * COLS + j + 1;

			indices.push_back(i0);
			indices.push_back(i1);
			indices.push_back(i2);

			indices.push_back(i1);
			indices.push_back(i3);
			indices.push_back(i2);
		}
	}
}

static void buildStretchConstraints(ConstraintHost& cons, const vector<glm::vec3>& h_pos) {
	cons.stretch.ver.clear();
	cons.stretch.k.clear();
	cons.stretch.l0.clear();
	// 가로 Stretch 제약
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS - 1; j++) {
			int a = i * COLS + j;
			int b = i * COLS + j + 1;

			glm::vec3 pa = h_pos[a];
			glm::vec3 pb = h_pos[b];

			float dx = pb.x - pa.x;
			float dy = pb.y - pa.y;
			float dz = pb.z - pa.z;

			float rest = sqrtf(dx * dx + dy * dy + dz * dz);

			cons.stretch.ver.push_back(make_int2(a, b));
			cons.stretch.k.push_back(K_STRETCH);
			cons.stretch.l0.push_back(rest);
		}
	}
	// 세로 stretch 제약
	for (int i = 0; i < ROWS - 1; i++) {
		for (int j = 0; j < COLS; j++) {
			int a = i * COLS + j;
			int b = (i + 1) * COLS + j;

			glm::vec3 pa = h_pos[a];
			glm::vec3 pb = h_pos[b];

			float dx = pb.x - pa.x;
			float dy = pb.y - pa.y;
			float dz = pb.z - pa.z;

			float rest = sqrtf(dx * dx + dy * dy + dz * dz);

			cons.stretch.ver.push_back(make_int2(a, b));
			cons.stretch.k.push_back(K_STRETCH);
			cons.stretch.l0.push_back(rest);
		}
	}
	// 대각선 stretch 제약
	for (int i = 0; i < ROWS - 1; i++) {
		for (int j = 0; j < COLS - 1; j++) {
			int p1 = i * COLS + j;
			int p2 = (i + 1) * COLS + j + 1;
			int p3 = i * COLS + j + 1;
			int p4 = (i + 1) * COLS + j;

			glm::vec3 pa = h_pos[p1];
			glm::vec3 pb = h_pos[p2];

			float dx = pb.x - pa.x;
			float dy = pb.y - pa.y;
			float dz = pb.z - pa.z;

			float rest = sqrtf(dx * dx + dy * dy + dz * dz);

			cons.stretch.ver.push_back(make_int2(p1, p2));
			cons.stretch.k.push_back(K_STRETCH);
			cons.stretch.l0.push_back(rest);

			pa = h_pos[p3];
			pb = h_pos[p4];

			dx = pb.x - pa.x;
			dy = pb.y - pa.y;
			dz = pb.z - pa.z;

			rest = sqrtf(dx * dx + dy * dy + dz * dz);

			cons.stretch.ver.push_back(make_int2(p3, p4));
			cons.stretch.k.push_back(K_STRETCH);
			cons.stretch.l0.push_back(rest);

		}
	}
}
static void buildBendingConstraints_AllSharedEdges(vector<unsigned int>& indices, vector<glm::vec3>& h_pos, ConstraintHost& cons) {
	cons.bending.ver.clear();
	cons.bending.k.clear();
	cons.bending.phi0.clear();

	unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> edgeMap;

	for (size_t t = 0; t < indices.size(); t += 3) {
		unsigned int tri[3] = { indices[t], indices[t + 1], indices[t + 2] };
		for (int e = 0; e < 3; e++) {
			unsigned int a = tri[e];
			unsigned int b = tri[(e + 1) % 3];
			unsigned int opp = tri[(e + 2) % 3];

			EdgeKey key = makeKey(a, b);
			auto it = edgeMap.find(key);

			if (it == edgeMap.end()) {
				edgeMap[key] = { opp };
 			}
			else {
				unsigned int c = it->second.opp;
				unsigned int d = opp;

				unsigned int v0 = key.a;
				unsigned int v1 = key.b;
				unsigned int v2 = c;
				unsigned int v3 = d;

				glm::vec3 p0 = h_pos[v0];
				glm::vec3 p1 = h_pos[v1];
				glm::vec3 p2 = h_pos[v2];
				glm::vec3 p3 = h_pos[v3];

				glm::vec3 e01 = p1 - p0;
				glm::vec3 n1 = glm::cross(e01, p2 - p0);
				glm::vec3 n2 = glm::cross(e01, p3 - p0);

				float n1Len = glm::length(n1);
				float n2Len = glm::length(n2);

				float phi0 = 0.0f;

				if (n1Len > 1e-8f && n2Len > 1e-8f) {
					float cval = glm::dot(n1, n2) / (n1Len * n2Len);
					cval = glm::clamp(cval, -1.0f, 1.0f);
					phi0 = glm::acos(cval);
				}
				cons.bending.ver.push_back(make_int4(v0, v1, v2, v3));
				cons.bending.k.push_back(K_BENDING);
				cons.bending.phi0.push_back(phi0);
			}
		}
	}
}

static void buildPerVertexConstraintLists(const ConstraintHost& cons, vector<unsigned int>& constraintsArray, vector<int>& constraintOffset, int vertexCount) {
	vector<vector<unsigned int>> perVertex(vertexCount);

	for (int i = 0; i < cons.stretch.ver.size(); i++) {
		int2 edge = cons.stretch.ver[i];
		perVertex[edge.x].push_back(packCons(Stretch, (unsigned int)i));
		perVertex[edge.y].push_back(packCons(Stretch, (unsigned int)i));
	}
	for (int i = 0; i < cons.bending.ver.size(); i++) {
		int4 tet = cons.bending.ver[i];
		perVertex[tet.x].push_back(packCons(Bending, (unsigned int)i));
		perVertex[tet.y].push_back(packCons(Bending, (unsigned int)i));
		perVertex[tet.z].push_back(packCons(Bending, (unsigned int)i));
		perVertex[tet.w].push_back(packCons(Bending, (unsigned int)i));
	}

	constraintsArray.clear();
	constraintOffset.assign(vertexCount + 1, 0);
	int running = 0;
	for (int i = 0; i < vertexCount; i++) {
		constraintOffset[i] = running;
		running += (int)perVertex[i].size();
	}
	constraintOffset[vertexCount] = running;
	for (int i = 0; i < vertexCount; i++) {
		for (int packed : perVertex[i]) {
			constraintsArray.push_back((unsigned int)packed);
		}
	}
}
static bool shareVertex(const int2& a, const int2& b) {
	return a.x == b.x || a.x == b.y || a.y == b.x || a.y == b.y;
}
static bool shareVertex(const int4& a, const int4& b) {
	int av[4] = { a.x, a.y, a.z, a.w };
	int bv[4] = { b.x, b.y, b.z, b.w };

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (av[i] == bv[j]) return true;
		}
	}
	return false;
}
static void buildStretchColorBatches(ConstraintHost& cons) {
	int n = (int)cons.stretch.ver.size();
	vector<vector<int>> groups;
	for (int i = 0; i < n; i++) {
		int chosenColor = -1;
		for (int c = 0; c < (int)groups.size(); c++) {
			bool conflict = false;
			for (int id : groups[c]) {
				if (shareVertex(cons.stretch.ver[i], cons.stretch.ver[id])) {
					conflict = true;
					break;
				}
			}
			if (!conflict) {
				chosenColor = c;
				break;
			}
		}
		if (chosenColor == -1) {
			groups.push_back({});
			chosenColor = (int)groups.size() - 1;
		}
		groups[chosenColor].push_back(i);
	}
	cons.stretch.color.constraintIds.clear();
	cons.stretch.color.colorOffset.clear();
	cons.stretch.color.colorOffset.push_back(0);
	for (int c = 0; c < (int)groups.size(); c++) {
		for (int id : groups[c]) {
			cons.stretch.color.constraintIds.push_back(id);
		}
		cons.stretch.color.colorOffset.push_back((int)cons.stretch.color.constraintIds.size());
	}
}
static void buildBendingColorBatches(ConstraintHost& cons) {
	int n = (int)cons.bending.ver.size();
	vector<vector<int>> groups;
	for (int i = 0; i < n; i++) {
		int chosenColor = -1;
		for (int c = 0; c < (int)groups.size(); c++) {
			bool conflict = false;
			for (int id : groups[c]) {
				if (shareVertex(cons.bending.ver[i], cons.bending.ver[id])) {
					conflict = true;
					break;
				}
			}
			if (!conflict) {
				chosenColor = c;
				break;
			}
		}
		if (chosenColor == -1) {
			groups.push_back({});
			chosenColor = (int)groups.size() - 1;
		}
		groups[chosenColor].push_back(i);
	}
	cons.bending.color.constraintIds.clear();
	cons.bending.color.colorOffset.clear();
	cons.bending.color.colorOffset.push_back(0);
	for (int c = 0; c < (int)groups.size(); c++) {
		for (int id : groups[c]) {
			cons.bending.color.constraintIds.push_back(id);
		}
		cons.bending.color.colorOffset.push_back((int)cons.bending.color.constraintIds.size());
	}
}

static void uploadColorBatch(const ColorBatchHost& host, ColorBatchDevice& device) {
	device.colorCount = (int)host.colorOffset.size() - 1;
	if (!host.constraintIds.empty()) {
		checkCuda(cudaMalloc(&device.constraintIds, sizeof(int) * host.constraintIds.size()), "cudaMalloc color.constraintIds failed");
		checkCuda(cudaMemcpy(device.constraintIds, host.constraintIds.data(), sizeof(int) * host.constraintIds.size(), cudaMemcpyHostToDevice), "cudaMemcpy color.constraintIds failed");
	}
	else {
		device.constraintIds = nullptr;
	}
	if (!host.colorOffset.empty()) {
		checkCuda(cudaMalloc(&device.colorOffset, sizeof(int) * host.colorOffset.size()), "cudaMalloc color.colorOffset failed");
		checkCuda(cudaMemcpy(device.colorOffset, host.colorOffset.data(), sizeof(int) * host.colorOffset.size(), cudaMemcpyHostToDevice), "cudaMemcpy color.colorOffset failed");
	}
	else {
		device.colorOffset = nullptr;
	}
}