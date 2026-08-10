
#include <iostream>
#include <cmath>
#include <cstdlib>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include "ClothBuilder.h"
#include "CudaUtils.cuh"
#include "CudaConstraintUtils.cuh"
#include "CommonSolver.cuh"
#include "XPBDSimulation.cuh"
#include "XPBDSolver.cuh"

namespace {
	__global__ void updateFloorKernel(float* d_floorVertices, float floorY) {
		if (blockIdx.x == 0 && threadIdx.x == 0) {
			d_floorVertices[1] = floorY;
			d_floorVertices[4] = floorY;
			d_floorVertices[7] = floorY;
			d_floorVertices[10] = floorY;
		}
	}

	void launchUpdateFloor(float* d_floorVertices, float floorY) {
		updateFloorKernel << <1, 1 >> > (d_floorVertices, floorY);
	}
	bool makeMouseRay(double mouseX, double mouseY, const glm::mat4& projection, const glm::mat4& view, unsigned int screenWidth, unsigned int screenHeight, glm::vec3& rayOrigin, glm::vec3& rayDirection) {
		if (screenWidth == 0 || screenHeight == 0) {
			return false;
		}
		float ndcX = 2.0f * static_cast<float>(mouseX) / static_cast<float>(screenWidth) - 1.0f;
		float ndcY = 1.0f - 2.0f * static_cast<float>(mouseY) / static_cast<float>(screenHeight);

		glm::mat4 inverseViewProjection = glm::inverse(projection * view);
		glm::vec4 nearPoint = inverseViewProjection * glm::vec4(ndcX, ndcY, -1.0f, 1.0f);
		glm::vec4 farPoint = inverseViewProjection * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);

		if (fabsf(nearPoint.w) < 1.0e-8f || fabsf(farPoint.w) < 1.0e-8f) {
			return false;
		}
		nearPoint /= nearPoint.w;
		farPoint /= farPoint.w;

		rayOrigin = glm::vec3(nearPoint);

		glm::vec3 direction = glm::vec3(farPoint) - rayOrigin;
		float directionLengthSquared = glm::dot(direction, direction);

		if (directionLengthSquared < 1.0e-12f) {
			return false;
		}
		rayDirection = direction / sqrtf(directionLengthSquared);
		return true;
	}
}
bool XPBDSimulation::beginMouseDrag(double mouseX, double mouseY, const glm::mat4& projection, const glm::mat4& view, unsigned int screenWidth, unsigned int screenHeight) {
	glm::vec3 rayOrigin;
	glm::vec3 rayDirection;

	if (!makeMouseRay(mouseX, mouseY, projection, view, screenWidth, screenHeight, rayOrigin, rayDirection)) {
		return false;
	}
	std::vector<float3> currentPositions(vertexCount);
	checkCuda(cudaMemcpy(currentPositions.data(), d_ver.p, sizeof(float3) * vertexCount, cudaMemcpyDeviceToHost), "cudaMemcpy mouse drag positions failed");
	
	float pickRadiusSquared = config.mousePickRadius * config.mousePickRadius;
	float bestDistanceSquared = pickRadiusSquared;
	float bestDepth = 0.0f;
	int bestVertex = -1;

	for (int i = 0; i < vertexCount; i++) {
		if (h_invM[i] == 0.0f) {
			continue;
		}
		const float3& p = currentPositions[i];
		glm::vec3 vertexPosition(p.x, p.y, p.z);

		float depth = glm::dot(vertexPosition - rayOrigin, rayDirection);
		if (depth <= 0.0f) {
			continue;
		}
		glm::vec3 closestPointOnRay = rayOrigin + depth * rayDirection;
		glm::vec3 difference = vertexPosition - closestPointOnRay;
		float distanceSquared = glm::dot(difference, difference);

		if (distanceSquared < bestDistanceSquared) {
			bestDistanceSquared = distanceSquared;
			bestDepth = depth;
			bestVertex = i;
		}
	}
	if (bestVertex < 0) {
		return false;
	}
	mouseDragActive = true;
	mouseDragVertex = bestVertex;
	mouseDragDepth = bestDepth;

	glm::vec3 target = rayOrigin + mouseDragDepth * rayDirection;
	mouseDragTarget = make_float3(target.x, target.y, target.z);

	return true;
}
void XPBDSimulation::updateMouseDrag(double mouseX, double mouseY, const glm::mat4& projection, const glm::mat4& view, unsigned int screenWidth, unsigned int screenHeight) {
	if (!mouseDragActive) {
		return;
	}
	glm::vec3 rayOrigin;
	glm::vec3 rayDirection;

	if (!makeMouseRay(mouseX, mouseY, projection, view, screenWidth, screenHeight, rayOrigin, rayDirection)) {
		return;
	}
	glm::vec3 target = rayOrigin + mouseDragDepth * rayDirection;
	mouseDragTarget = make_float3(target.x, target.y, target.z);
}
void XPBDSimulation::endMouseDrag() {
	mouseDragActive = false;
	mouseDragVertex = -1;
	mouseDragDepth = 0.0f;
}
XPBDSimulation::XPBDSimulation(const SimulationConfig& config_) : config(config_) {
	vertexCount = config.rows * config.cols;
	currentFloorY = config.floorBaseY;
}

bool XPBDSimulation::initialize() {
	checkCuda(cudaGLSetGLDevice(0), "cudaGLSetGLDevice failed");

	if (!initializeHostData()) {
		return false;
	}

	if (!initializeClothRenderResources()) {
		return false;
	}

	if (!initializeDeviceData()) {
		return false;
	}

	if (!initializeFloorResources()) {
		return false;
	}

	if (!initializeSimulationBuffers()) {
		return false;
	}
	return true;
}

bool XPBDSimulation::initializeHostData() {
	buildClothPositions(h_pos, h_v, h_p, h_dp, h_invM, config.rows, config.cols);

	buildStretchConstraints(h_cons, h_pos, config.rows, config.cols, config.stretchK);

	buildTriangleIndices(triangleIndices, config.rows, config.cols);

	buildVertexTriangleAdjacency(triangleIndices, vertexCount, h_vertTriArray, h_vertTriOffset);

	buildBendingConstraints_AllSharedEdges(triangleIndices, h_pos, h_cons, config.bendingK);

	buildStretchColorBatches(h_cons);
	buildBendingColorBatches(h_cons);

	buildPerVertexConstraintLists(h_cons, h_constraintsArray, h_constraintOffset, vertexCount);

	return true;
}

bool XPBDSimulation::initializeClothRenderResources() {
	glGenVertexArrays(1, &clothVAO);
	glBindVertexArray(clothVAO);

	glGenBuffers(1, &clothVBO);
	glBindBuffer(GL_ARRAY_BUFFER, clothVBO);
	glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &clothEBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, clothEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, triangleIndices.size() * sizeof(unsigned int), triangleIndices.data(), GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);

	checkCuda(cudaGraphicsGLRegisterBuffer(&cudaVBO, clothVBO, cudaGraphicsMapFlagsNone), "cudaGraphicsGLRegisterBuffer failed");

	checkCuda(cudaGraphicsMapResources(1, &cudaVBO), "cudaGraphicsMapResources(init) failed");

	size_t mappedSize = 0;
	checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_mappedVboPos, &mappedSize, cudaVBO), "cudaGraphicsResourceGetMappedPointer(init) failed");
	checkCuda(cudaMemcpy(d_mappedVboPos, h_pos.data(), sizeof(float3) * vertexCount, cudaMemcpyHostToDevice), "Initial copy to VBO failed");

	checkCuda(cudaGraphicsUnmapResources(1, &cudaVBO), "cudaGraphicsUnmapResources(init) failed");

	d_mappedVboPos = nullptr;

	return true;
}

bool XPBDSimulation::initializeDeviceData() {
	checkCuda(cudaMalloc(&d_triangleIndices, sizeof(unsigned int) * triangleIndices.size()), "cudaMalloc d_triangleIndices failed");
	checkCuda(cudaMemcpy(d_triangleIndices, triangleIndices.data(), sizeof(unsigned int) * triangleIndices.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_triangleIndices failed");

	checkCuda(cudaMalloc(&d_vertTriArray, sizeof(int) * h_vertTriArray.size()), "cudaMalloc d_vertTriArray failed");
	checkCuda(cudaMalloc(&d_vertTriOffset, sizeof(int) * h_vertTriOffset.size()), "cudaMalloc d_vertTriOffset failed");

	checkCuda(cudaMemcpy(d_vertTriArray, h_vertTriArray.data(), sizeof(int) * h_vertTriArray.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_vertTriArray failed");
	checkCuda(cudaMemcpy(d_vertTriOffset, h_vertTriOffset.data(), sizeof(int) * h_vertTriOffset.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_vertTriOffset failed");

	d_ver.pos = nullptr;
	d_ver.N = vertexCount;
	d_ver.constraintNum = static_cast<int>(h_constraintsArray.size());

	checkCuda(cudaMalloc(&d_ver.v, sizeof(float3) * vertexCount), "cudaMalloc d_ver.v failed");
	checkCuda(cudaMalloc(&d_ver.p, sizeof(float3) * vertexCount), "cudaMalloc d_ver.p failed");
	checkCuda(cudaMalloc(&d_ver.dp, sizeof(float3) * vertexCount), "cudaMalloc d_ver.dp failed");
	checkCuda(cudaMalloc(&d_ver.dpCount, sizeof(int) * vertexCount), "cudaMalloc d_ver.dpCount failed");
	checkCuda(cudaMalloc(&d_ver.invM, sizeof(float) * vertexCount), "cudaMalloc d_ver.invM failed");
	checkCuda(cudaMalloc(&d_ver.constraintsArray, sizeof(unsigned int) * h_constraintsArray.size()), "cudaMalloc d_ver.constraintsArray failed");
	checkCuda(cudaMalloc(&d_ver.constraintOffset, sizeof(int) * (vertexCount + 1)), "cudaMalloc d_ver.constraintOffset failed");

	checkCuda(cudaMemcpy(d_ver.v, h_v.data(), sizeof(float3) * vertexCount, cudaMemcpyHostToDevice), "cudaMemcpy d_ver.v failed");
	checkCuda(cudaMemcpy(d_ver.p, h_p.data(), sizeof(float3) * vertexCount, cudaMemcpyHostToDevice), "cudaMemcpy d_ver.p failed");
	checkCuda(cudaMemcpy(d_ver.dp, h_dp.data(), sizeof(float3) * vertexCount, cudaMemcpyHostToDevice), "cudaMemcpy d_ver.dp failed");
	checkCuda(cudaMemset(d_ver.dpCount, 0, sizeof(int) * vertexCount), "cudaMemset d_ver.dpCount failed");
	checkCuda(cudaMemcpy(d_ver.invM, h_invM.data(), sizeof(float) * vertexCount, cudaMemcpyHostToDevice), "cudaMemcpy d_ver.invM failed");
	checkCuda(cudaMemcpy(d_ver.constraintsArray, h_constraintsArray.data(), sizeof(unsigned int) * h_constraintsArray.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_ver.constraintsArray failed");
	checkCuda(cudaMemcpy(d_ver.constraintOffset, h_constraintOffset.data(), sizeof(int) * (vertexCount + 1), cudaMemcpyHostToDevice), "cudaMemcpy d_ver.constraintOffset failed");

	d_cons.stretch.color.constraintIds = nullptr;
	d_cons.stretch.color.colorOffset = nullptr;
	d_cons.stretch.color.colorCount = 0;

	d_cons.bending.color.constraintIds = nullptr;
	d_cons.bending.color.colorOffset = nullptr;
	d_cons.bending.color.colorCount = 0;

	d_cons.collision.tri = nullptr;
	d_cons.collision.ver = nullptr;
	d_cons.collision.k = nullptr;
	d_cons.collision.lambda = nullptr;
	d_cons.collision.compliance = nullptr;
	d_cons.collision.thickness = nullptr;
	d_cons.collision.q = nullptr;
	d_cons.collision.normal = nullptr;
	d_cons.collision.colliderVelocity = nullptr;
	d_cons.collision.n = nullptr;
	d_cons.collision.capacity = vertexCount * 8;

	checkCuda(cudaMalloc(&d_cons.collision.tri, sizeof(int3) * d_cons.collision.capacity), "cudaMalloc collision.tri failed");
	checkCuda(cudaMalloc(&d_cons.collision.ver, sizeof(int) * d_cons.collision.capacity), "cudaMalloc collision.ver failed");
	checkCuda(cudaMalloc(&d_cons.collision.k, sizeof(float) * d_cons.collision.capacity), "cudaMalloc collision.k failed");
	checkCuda(cudaMalloc(&d_cons.collision.compliance, sizeof(float) * d_cons.collision.capacity), "cudaMalloc collision.compliance failed");
	checkCuda(cudaMalloc(&d_cons.collision.lambda, sizeof(float) * d_cons.collision.capacity), "cudaMalloc collision.lambda failed");
	checkCuda(cudaMalloc(&d_cons.collision.thickness, sizeof(float) * d_cons.collision.capacity), "cudaMalloc collision.thickness failed");
	checkCuda(cudaMalloc(&d_cons.collision.q, sizeof(float3) * d_cons.collision.capacity), "cudaMalloc collision.q failed");
	checkCuda(cudaMalloc(&d_cons.collision.normal, sizeof(float3) * d_cons.collision.capacity), "cudaMalloc collision.normal failed");
	checkCuda(cudaMalloc(&d_cons.collision.colliderVelocity, sizeof(float3) * d_cons.collision.capacity), "cudaMalloc collision.colliderVelocity failed");
	checkCuda(cudaMalloc(&d_cons.collision.n, sizeof(int)), "cudaMalloc collision.n failed");

	int zero = 0;
	checkCuda(cudaMemcpy(d_cons.collision.n, &zero, sizeof(int), cudaMemcpyHostToDevice), "init collision.n failed");
	checkCuda(cudaMemset(d_cons.collision.compliance, 0, sizeof(float) * d_cons.collision.capacity), "cudaMemset collision.compliance failed");
	checkCuda(cudaMemset(d_cons.collision.lambda, 0, sizeof(float) * d_cons.collision.capacity), "cudaMemset collision.lambda failed");

	d_cons.selfCollision.tri = nullptr;
	d_cons.selfCollision.ver = nullptr;
	d_cons.selfCollision.k = nullptr;
	d_cons.selfCollision.compliance = nullptr;
	d_cons.selfCollision.lambda = nullptr;
	d_cons.selfCollision.thickness = nullptr;
	d_cons.selfCollision.q = nullptr;
	d_cons.selfCollision.normal = nullptr;
	d_cons.selfCollision.n = nullptr;
	d_cons.selfCollision.capacity = vertexCount * 8;

	checkCuda(cudaMalloc(&d_cons.selfCollision.tri, sizeof(int3) * d_cons.selfCollision.capacity), "cudaMalloc selfCollision.tri failed");
	checkCuda(cudaMalloc(&d_cons.selfCollision.ver, sizeof(int) * d_cons.selfCollision.capacity), "cudaMalloc selfCollision.ver failed");
	checkCuda(cudaMalloc(&d_cons.selfCollision.k, sizeof(float) * d_cons.selfCollision.capacity), "cudaMalloc selfCollision.k failed");
	checkCuda(cudaMalloc(&d_cons.selfCollision.compliance, sizeof(float) * d_cons.selfCollision.capacity), "cudaMalloc selfCollision.compliance failed");
	checkCuda(cudaMalloc(&d_cons.selfCollision.lambda, sizeof(float) * d_cons.selfCollision.capacity), "cudaMalloc selfCollision.lambda failed");
	checkCuda(cudaMalloc(&d_cons.selfCollision.thickness, sizeof(float) * d_cons.selfCollision.capacity), "cudaMalloc selfCollision.thickness failed");
	checkCuda(cudaMalloc(&d_cons.selfCollision.q, sizeof(float3) * d_cons.selfCollision.capacity), "cudaMalloc selfCollision.q failed");
	checkCuda(cudaMalloc(&d_cons.selfCollision.normal, sizeof(float3) * d_cons.selfCollision.capacity), "cudaMalloc selfCollision.normal failed");
	checkCuda(cudaMalloc(&d_cons.selfCollision.n, sizeof(int)), "cudaMalloc selfCollision.n failed");
	
	checkCuda(cudaMemset(d_cons.selfCollision.compliance, 0, sizeof(float) * d_cons.selfCollision.capacity), "cudaMemset selfCollision.compliance failed");
	checkCuda(cudaMemset(d_cons.selfCollision.lambda, 0, sizeof(float) * d_cons.selfCollision.capacity), "cudaMemset selfCollision.lambda failed");

	int zeroSelfCollision = 0;
	checkCuda(cudaMemcpy(d_cons.selfCollision.n, &zeroSelfCollision, sizeof(int), cudaMemcpyHostToDevice), "init selfCollision.n failed");

	d_cons.stretch.n = static_cast<int>(h_cons.stretch.ver.size());
	d_cons.bending.n = static_cast<int>(h_cons.bending.ver.size());
	d_cons.stretch.ver = nullptr;
	d_cons.stretch.k = nullptr;
	d_cons.stretch.l0 = nullptr;
	d_cons.stretch.lambda = nullptr;

	d_cons.bending.ver = nullptr;
	d_cons.bending.k = nullptr;
	d_cons.bending.phi0 = nullptr;
	d_cons.bending.lambda = nullptr;

	checkCuda(cudaMalloc(&d_cons.stretch.ver, sizeof(int2) * d_cons.stretch.n), "cudaMalloc stretch.ver failed");
	checkCuda(cudaMalloc(&d_cons.stretch.k, sizeof(float) * d_cons.stretch.n), "cudaMalloc stretch.k failed");
	checkCuda(cudaMalloc(&d_cons.stretch.l0, sizeof(float) * d_cons.stretch.n), "cudaMalloc stretch.l0 failed");
	checkCuda(cudaMalloc(&d_cons.stretch.lambda, sizeof(float) * d_cons.stretch.n), "cudaMalloc stretch.lambda failed");

	if (d_cons.stretch.n > 0) {
		checkCuda(cudaMemcpy(d_cons.stretch.ver, h_cons.stretch.ver.data(), sizeof(int2) * d_cons.stretch.n, cudaMemcpyHostToDevice), "cudaMemcpy stretch.ver failed");
		checkCuda(cudaMemcpy(d_cons.stretch.k, h_cons.stretch.k.data(), sizeof(float) * d_cons.stretch.n, cudaMemcpyHostToDevice), "cudaMemcpy stretch.k failed");
		checkCuda(cudaMemcpy(d_cons.stretch.l0, h_cons.stretch.l0.data(), sizeof(float) * d_cons.stretch.n, cudaMemcpyHostToDevice), "cudaMemcpy stretch.l0 failed");
		checkCuda(cudaMemset(d_cons.stretch.lambda, 0, sizeof(float) * d_cons.stretch.n), "cudaMemset stretch.lambda failed");
	}

	checkCuda(cudaMalloc(&d_cons.bending.ver, sizeof(int4) * d_cons.bending.n), "cudaMalloc bending.ver failed");
	checkCuda(cudaMalloc(&d_cons.bending.k, sizeof(float) * d_cons.bending.n), "cudaMalloc bending.k failed");
	checkCuda(cudaMalloc(&d_cons.bending.phi0, sizeof(float) * d_cons.bending.n), "cudaMalloc bending.phi0 failed");
	checkCuda(cudaMalloc(&d_cons.bending.lambda, sizeof(float) * d_cons.bending.n), "cudaMalloc bending.lambda failed");
	if (d_cons.bending.n > 0) {
		checkCuda(cudaMemcpy(d_cons.bending.ver, h_cons.bending.ver.data(), sizeof(int4) * d_cons.bending.n, cudaMemcpyHostToDevice), "cudaMemcpy bending.ver failed");
		checkCuda(cudaMemcpy(d_cons.bending.k, h_cons.bending.k.data(), sizeof(float) * d_cons.bending.n, cudaMemcpyHostToDevice), "cudaMemcpy bending.k failed");
		checkCuda(cudaMemcpy(d_cons.bending.phi0, h_cons.bending.phi0.data(), sizeof(float) * d_cons.bending.n, cudaMemcpyHostToDevice), "cudaMemcpy bending.phi0 failed");
		checkCuda(cudaMemset(d_cons.bending.lambda, 0, sizeof(float) * d_cons.bending.n), "cudaMemset bending.lambda failed");
	}

	uploadColorBatch(h_cons.stretch.color, d_cons.stretch.color);
	uploadColorBatch(h_cons.bending.color, d_cons.bending.color);

	checkCuda(cudaMalloc(&d_damp.poscm, sizeof(float3)), "cudaMalloc damp.poscm failed");
	checkCuda(cudaMalloc(&d_damp.vcm, sizeof(float3)), "cudaMalloc damp.vcm failed");
	checkCuda(cudaMalloc(&d_damp.omega, sizeof(float3)), "cudaMalloc damp.omega failed");
	checkCuda(cudaMalloc(&d_damp.angularMomentum, sizeof(float3)), "cudaMalloc damp.angularMomentum failed");
	checkCuda(cudaMalloc(&d_damp.inertia, sizeof(float) * 9), "cudaMalloc damp.inertia failed");

	return true;
}

bool XPBDSimulation::initializeFloorResources() {
	float floorVertices[] = {
		-2.0f, currentFloorY, -2.0f,
		 2.0f, currentFloorY, -2.0f,
		 2.0f, currentFloorY,  2.0f,
		-2.0f, currentFloorY,  2.0f
	};

	unsigned int floorIndices[] = {
		0, 2, 1,
		0, 3, 2
	};

	checkCuda(cudaMalloc(&d_floorVertices, sizeof(floorVertices)), "cudaMalloc d_floorVertices failed");
	checkCuda(cudaMalloc(&d_prevFloorVertices, sizeof(floorVertices)), "cudaMalloc d_prevFloorVertices failed");
	checkCuda(cudaMalloc(&d_floorIndices, sizeof(floorIndices)), "cudaMalloc d_floorIndices failed");

	checkCuda(cudaMemcpy(d_floorVertices, floorVertices, sizeof(floorVertices), cudaMemcpyHostToDevice), "cudaMemcpy d_floorVertices failed");
	checkCuda(cudaMemcpy(d_prevFloorVertices, floorVertices, sizeof(floorVertices), cudaMemcpyHostToDevice), "cudaMemcpy d_prevFloorVertices failed");
	checkCuda(cudaMemcpy(d_floorIndices, floorIndices, sizeof(floorIndices), cudaMemcpyHostToDevice), "cudaMemcpy d_floorIndices failed");

	vertexSet.push_back(d_floorVertices);
	prevVertexSet.push_back(d_prevFloorVertices);
	indexSet.push_back(d_floorIndices);
	indexSetN.push_back(6);

	glGenVertexArrays(1, &floorVAO);
	glBindVertexArray(floorVAO);

	glGenBuffers(1, &floorVBO);
	glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(floorVertices), floorVertices, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &floorEBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, floorEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(floorIndices), floorIndices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(0);

	return true;
}

bool XPBDSimulation::initializeSimulationBuffers() {
	checkCuda(cudaMalloc(&d_selfPairs, sizeof(int2) * config.maxSelfPairs), "d_selfPairs malloc");
	checkCuda(cudaMalloc(&d_selfPairCount, sizeof(int)), "d_selfPairCount malloc");

	std::vector<int4> h_selfTris;

	for (size_t t = 0; t + 2 < triangleIndices.size(); t += 3) {
		int i0 = static_cast<int>(triangleIndices[t + 0]);
		int i1 = static_cast<int>(triangleIndices[t + 1]);
		int i2 = static_cast<int>(triangleIndices[t + 2]);

		h_selfTris.push_back({ i0, i1, i2, 0 });
	}

	triN = static_cast<int>(h_selfTris.size());

	if (triN > 0) {
		checkCuda(cudaMalloc(&d_selfTris, sizeof(int4) * triN), "selfTris malloc");
		checkCuda(cudaMemcpy(d_selfTris, h_selfTris.data(), sizeof(int4) * triN, cudaMemcpyHostToDevice), "selfTris memcpy");
	}

	glm::vec3 wind = glm::vec3(rand() % 2, 0, rand() % 2);
	glm::vec3 gravity = glm::vec3(0.0f, -9.8f, 0.0f);
	//h_forces.push_back(wind);
	h_forces.push_back(gravity);

	checkCuda(cudaMalloc(&d_forces, sizeof(float3) * h_forces.size()), "cudaMalloc d_forces failed");
	checkCuda(cudaMemcpy(d_forces, h_forces.data(), sizeof(float3) * h_forces.size(), cudaMemcpyHostToDevice), "cudaMemcpy d_forces failed");

	checkCuda(cudaMalloc(&d_totalMass, sizeof(float)), "totalMass malloc");
	checkCuda(cudaMalloc(&d_totalForce, sizeof(float3)), "cudaMalloc d_totalForce failed");
	if (d_totalForce == nullptr) {
		throw std::runtime_error("cudaMalloc succeeded but d_totalForce is null");
	}
	d_damp.partialCount = (vertexCount + DAMPING_REDUCTION_THREADS - 1) / DAMPING_REDUCTION_THREADS;
	checkCuda(cudaMalloc(&d_damp.centerPartials, sizeof(DampingCenterPartial) * d_damp.partialCount), "cudaMalloc damp.centerPartials failed");
	checkCuda(cudaMalloc(&d_damp.angularPartials, sizeof(DampingAngularPartial) * d_damp.partialCount), "cudaMalloc damp.angularPartials failed");

	//checkCuda(cudaEventCreate(&simStartEvent), "cudaEventCreate simStartEvent failed");
	//checkCuda(cudaEventCreate(&simEndEvent), "cudaEventCreate simEndEvent failed");

	checkCuda(cudaMalloc(&d_gridHashes, sizeof(unsigned int) * vertexCount), "cudaMalloc d_gridHashes failed");
	checkCuda(cudaMalloc(&d_gridIndices, sizeof(int) * vertexCount), "cudaMalloc d_gridIndices failed");
	checkCuda(cudaMalloc(&d_cellStart, sizeof(int) * config.gridCapacity), "cudaMalloc d_cellStart failed");
	checkCuda(cudaMalloc(&d_cellEnd, sizeof(int) * config.gridCapacity), "cudaMalloc d_cellEnd failed");

	return true;
}

void XPBDSimulation::step(float currentTime) {
	mapClothVBO();

	simulateSubsteps(currentTime);

	checkCuda(cudaDeviceSynchronize(),"XPBD GPU execution failed");

	unmapClothVBO();
}

void XPBDSimulation::mapClothVBO() {
	checkCuda(cudaGraphicsMapResources(1, &cudaVBO), "cudaGraphicsMapResources(frame) failed");

	size_t mappedSize = 0;
	checkCuda(cudaGraphicsResourceGetMappedPointer((void**)&d_mappedVboPos, &mappedSize, cudaVBO), "cudaGraphicsResourceGetMappedPointer(frame) failed");

	d_ver.pos = d_mappedVboPos;
}
void XPBDSimulation::solveOneSubstep(float dtSub, float currentTime) {
	if (d_totalForce == nullptr) {
		throw std::runtime_error("d_totalForce is null before xpbd::solve");
	}
	xpbd::solve(d_ver, d_cons, d_damp, constraintIterationGraph, vertexSet, prevVertexSet, indexSet, indexSetN, d_gridIndices, d_cellStart, d_cellEnd, d_gridHashes, d_totalMass, d_totalForce, d_selfTris, d_vertTriArray, d_vertTriOffset, config.selfCollisionRadius, config.selfCollisionThickness, config.selfCollisionK, config.gridCapacity, d_forces, config.stretchDamping, config.bendingDamping, dtSub, currentTime, config.iterationCount, static_cast<int>(h_forces.size()), vertexCount, h_cons.stretch.color.colorOffset, h_cons.bending.color.colorOffset, config.friction, config.restitution ,mouseDragActive, mouseDragVertex, mouseDragTarget);
}
void XPBDSimulation::unmapClothVBO() {
	checkCuda(cudaGraphicsUnmapResources(1, &cudaVBO), "cudaGraphicsUnmapResources(frame) failed");

	d_mappedVboPos = nullptr;
	d_ver.pos = nullptr;
}

void XPBDSimulation::simulateSubsteps(float currentTime) {
	float dtSub = config.timestep / static_cast<float>(config.substep);

	//checkCuda(cudaEventRecord(simStartEvent), "cudaEventRecord simStartEvent failed");

	for (int s = 0; s < config.substep; s++) {
		float subAlpha = static_cast<float>(s + 1) / static_cast<float>(config.substep);
		float motionTime = currentTime + subAlpha * config.timestep;

		simTime += dtSub;

		currentFloorY = config.floorBaseY - config.floorAmplitude * std::cos(motionTime * config.floorSpeed);

		updateFloorDevice(currentFloorY);

		solveOneSubstep(dtSub, currentTime);
	}

	updateFloorRenderBuffer(currentFloorY);
}

void XPBDSimulation::updateFloorDevice(float floorY) {
	checkCuda(cudaMemcpy(d_prevFloorVertices, d_floorVertices, sizeof(float) * 12, cudaMemcpyDeviceToDevice), "cudaMemcpy d_floorVertices to d_prevFloorVertices failed");

	float floorVertices[] = {
		-2.0f, floorY, -2.0f,
		 2.0f, floorY, -2.0f,
		 2.0f, floorY,  2.0f,
		-2.0f, floorY,  2.0f
	};

	checkCuda(cudaMemcpy(d_floorVertices, floorVertices, sizeof(floorVertices), cudaMemcpyHostToDevice), "cudaMemcpy moving floor to device failed");
}

void XPBDSimulation::updateFloorRenderBuffer(float floorY) {
	float floorVertices[] = {
		-2.0f, floorY, -2.0f,
		 2.0f, floorY, -2.0f,
		 2.0f, floorY,  2.0f,
		-2.0f, floorY,  2.0f
	};

	glBindBuffer(GL_ARRAY_BUFFER, floorVBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(floorVertices), floorVertices);
}

void XPBDSimulation::render(RenderContext& renderContext) {
	renderCloth(renderContext);
	renderFloor(renderContext);
}

void XPBDSimulation::renderCloth(RenderContext& renderContext) {
	Shader& clothShader = renderContext.getClothShader();

	glm::mat4 projection = renderContext.getProjection();
	glm::mat4 view = renderContext.getView();
	glm::mat4 model = glm::mat4(1.0f);

	clothShader.use();
	clothShader.setMat4("projection", projection);
	clothShader.setMat4("view", view);
	clothShader.setMat4("model", model);

	glBindVertexArray(clothVAO);
	glDrawElements(GL_LINES, static_cast<GLsizei>(triangleIndices.size()), GL_UNSIGNED_INT, 0);
}

void XPBDSimulation::renderFloor(RenderContext& renderContext) {
	Shader& floorShader = renderContext.getFloorShader();

	glm::mat4 projection = renderContext.getProjection();
	glm::mat4 view = renderContext.getView();
	glm::mat4 model = glm::mat4(1.0f);

	floorShader.use();
	floorShader.setMat4("projection", projection);
	floorShader.setMat4("view", view);
	floorShader.setMat4("model", model);

	glBindVertexArray(floorVAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void XPBDSimulation::release() {
	releaseCudaResources();
	releaseOpenGLResources();
}

void XPBDSimulation::releaseCudaResources() {
	constraintIterationGraph.release();

	//if (simStartEvent != nullptr) {
	//	cudaEventDestroy(simStartEvent);
	//	simStartEvent = nullptr;
	//}

	//if (simEndEvent != nullptr) {
	//	cudaEventDestroy(simEndEvent);
	//	simEndEvent = nullptr;
	//}

	if (cudaVBO != nullptr) {
		cudaGraphicsUnregisterResource(cudaVBO);
		cudaVBO = nullptr;
	}

	cudaFree(d_totalMass);
	cudaFree(d_forces);
	cudaFree(d_totalForce);


	cudaFree(d_selfPairs);
	cudaFree(d_selfPairCount);
	cudaFree(d_selfTris);

	cudaFree(d_vertTriArray);
	cudaFree(d_vertTriOffset);
	cudaFree(d_triangleIndices);

	cudaFree(d_gridHashes);
	cudaFree(d_gridIndices);
	cudaFree(d_cellStart);
	cudaFree(d_cellEnd);

	cudaFree(d_damp.poscm);
	cudaFree(d_damp.vcm);
	cudaFree(d_damp.omega);
	cudaFree(d_damp.angularMomentum);
	cudaFree(d_damp.inertia);
	cudaFree(d_damp.centerPartials);
	cudaFree(d_damp.angularPartials);

	cudaFree(d_cons.collision.tri);
	cudaFree(d_cons.collision.ver);
	cudaFree(d_cons.collision.k);
	cudaFree(d_cons.collision.compliance);
	cudaFree(d_cons.collision.lambda);
	cudaFree(d_cons.collision.thickness);
	cudaFree(d_cons.collision.q);
	cudaFree(d_cons.collision.normal);
	cudaFree(d_cons.collision.colliderVelocity);
	cudaFree(d_cons.collision.n);

	cudaFree(d_cons.selfCollision.tri);
	cudaFree(d_cons.selfCollision.ver);
	cudaFree(d_cons.selfCollision.k);
	cudaFree(d_cons.selfCollision.compliance);
	cudaFree(d_cons.selfCollision.lambda);
	cudaFree(d_cons.selfCollision.thickness);
	cudaFree(d_cons.selfCollision.q);
	cudaFree(d_cons.selfCollision.normal);
	cudaFree(d_cons.selfCollision.n);

	cudaFree(d_cons.stretch.ver);
	cudaFree(d_cons.stretch.k);
	cudaFree(d_cons.stretch.l0);
	cudaFree(d_cons.stretch.lambda);

	cudaFree(d_cons.bending.ver);
	cudaFree(d_cons.bending.k);
	cudaFree(d_cons.bending.phi0);
	cudaFree(d_cons.bending.lambda);

	cudaFree(d_cons.stretch.color.constraintIds);
	cudaFree(d_cons.stretch.color.colorOffset);

	cudaFree(d_cons.bending.color.constraintIds);
	cudaFree(d_cons.bending.color.colorOffset);

	cudaFree(d_ver.v);
	cudaFree(d_ver.p);
	cudaFree(d_ver.dp);
	cudaFree(d_ver.dpCount);
	cudaFree(d_ver.invM);
	cudaFree(d_ver.constraintsArray);
	cudaFree(d_ver.constraintOffset);

	cudaFree(d_floorVertices);
	cudaFree(d_prevFloorVertices);
	cudaFree(d_floorIndices);
}

void XPBDSimulation::releaseOpenGLResources() {
	if (clothVAO != 0) {
		glDeleteVertexArrays(1, &clothVAO);
		clothVAO = 0;
	}

	if (clothVBO != 0) {
		glDeleteBuffers(1, &clothVBO);
		clothVBO = 0;
	}

	if (clothEBO != 0) {
		glDeleteBuffers(1, &clothEBO);
		clothEBO = 0;
	}

	if (floorVAO != 0) {
		glDeleteVertexArrays(1, &floorVAO);
		floorVAO = 0;
	}

	if (floorVBO != 0) {
		glDeleteBuffers(1, &floorVBO);
		floorVBO = 0;
	}

	if (floorEBO != 0) {
		glDeleteBuffers(1, &floorEBO);
		floorEBO = 0;
	}
}