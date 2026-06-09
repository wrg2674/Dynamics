#pragma once

#include <vector>

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <glm/glm.hpp>


#include "SimulationConfig.h"
#include "RenderContext.h"
#include "Vertex.h"
#include "Constraint.h"
#include "Damping.h"
#include "CudaConstraintUtils.cuh"


//__global__ void updateFloorKernel(float* d_floorVertices, float floorY);
//void launchUpdateFloor(float* d_floorVertices, float floorY);


class XPBDSimulation {
public:
	explicit XPBDSimulation(const SimulationConfig& config);

	bool initialize();
	void step(float currentTime);
	void render(RenderContext& renderContext);
	void release();

private:
	bool initializeHostData();
	bool initializeClothRenderResources();
	bool initializeDeviceData();
	bool initializeFloorResources();
	bool initializeSimulationBuffers();

	void mapClothVBO();
	void unmapClothVBO();

	void simulateSubsteps(float currentTime);
	void solveOneSubstep(float dtSub, float currentTime);
	void updateFloorDevice(float floorY);
	void updateFloorRenderBuffer(float floorY);

	void renderCloth(RenderContext& renderContext);
	void renderFloor(RenderContext& renderContext);

	void releaseCudaResources();
	void releaseOpenGLResources();

private:
	CudaConstraintGraph constraintIterationGraph;
	SimulationConfig config;

	int vertexCount = 0;

	std::vector<glm::vec3> h_pos;
	std::vector<glm::vec3> h_v;
	std::vector<glm::vec3> h_p;
	std::vector<glm::vec3> h_dp;
	std::vector<float> h_invM;

	std::vector<unsigned int> triangleIndices;
	std::vector<unsigned int> h_constraintsArray;
	std::vector<int> h_constraintOffset;

	std::vector<int> h_vertTriArray;
	std::vector<int> h_vertTriOffset;

	ConstraintHost h_cons;

	GLuint clothVAO = 0;
	GLuint clothVBO = 0;
	GLuint clothEBO = 0;

	GLuint floorVAO = 0;
	GLuint floorVBO = 0;
	GLuint floorEBO = 0;

	cudaGraphicsResource* cudaVBO = nullptr;

	VertexDevice d_ver{};
	ConstraintDevice d_cons{};
	DampingDevice d_damp{};

	unsigned int* d_triangleIndices = nullptr;

	int* d_vertTriArray = nullptr;
	int* d_vertTriOffset = nullptr;

	int2* d_selfPairs = nullptr;
	int* d_selfPairCount = nullptr;

	int4* d_selfTris = nullptr;
	int triN = 0;

	float* d_floorVertices = nullptr;
	float* d_prevFloorVertices = nullptr;
	unsigned int* d_floorIndices = nullptr;

	std::vector<float*> vertexSet;
	std::vector<float*> prevVertexSet;
	std::vector<unsigned int*> indexSet;
	std::vector<int> indexSetN;

	float3* d_forces = nullptr;
	std::vector<glm::vec3> h_forces;

	float* d_totalMass = nullptr;
	float3* d_totalForce = nullptr;

	unsigned int* d_gridHashes = nullptr;
	int* d_gridIndices = nullptr;
	int* d_cellStart = nullptr;
	int* d_cellEnd = nullptr;

	cudaEvent_t simStartEvent = nullptr;
	cudaEvent_t simEndEvent = nullptr;

	float3* d_mappedVboPos = nullptr;

	float simTime = 0.0f;
	float currentFloorY = -1.5f;

	double simTimeSumMs = 0.0;
	int simFrameCount = 0;
};
