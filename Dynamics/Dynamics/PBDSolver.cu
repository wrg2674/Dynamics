#include "PBDSolver.cuh"
#include "CommonSolver.cuh"

#include <device_launch_parameters.h>

__device__ void calcDeltaP(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep) {
	float3 gradient = make_float3(0, 0, 0);

	calcGradient(ver, cons, verIndex, consIndex, tstep, type, gradient);

	float s = calcScale(ver, cons, verIndex, consIndex, type, tstep);
	for (int i = 0; i < 3; i++) {
		set(ver.dp[verIndex], i, -s * ver.invM[verIndex] * get(gradient, i));
	}
}
__device__ void projectionFunction(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep, int ns) {
	calcDeltaP(ver, cons, verIndex, consIndex, type, tstep);
	float weight = 0;
	switch (type) {
	case Stretch:
		weight = 1.0 - pow((1.0 - cons.stretch.k[consIndex]), 1.0 / ns);
		break;
	case Bending:
		weight = 1.0 - pow((1.0 - cons.bending.k[consIndex]), 1.0 / ns);
		break;
	}
	// GS 스타일의 즉시 업데이트는 제약사항 단위의 것을 의미하는 것이지 
	// 한 제약사항 내에서 각 정점마다 즉시 업데이트를 하면 안됨.
	for (int i = 0; i < 3; i++) {
		set(ver.p[verIndex], i, get(ver.p[verIndex], i) + weight * get(ver.dp[verIndex], i));
	}
}

__device__ float calcScale(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, Type type, float tstep) {
	float result = 0;
	float upper = 0;
	float lower = 0;

	switch (type) {
	case Stretch: {
		upper = calcStretch(ver, cons, consIndex);
		int2 verIndexList = cons.stretch.ver[consIndex];
		for (int i = 0; i < 2; i++) {
			float3 gradient = make_float3(0, 0, 0);
			calcGradient(ver, cons, get(verIndexList, i), consIndex, tstep, type, gradient);
			float norm = norm2(gradient);
			lower += ver.invM[get(verIndexList, i)] * norm;
		}
		break;
	}
	case Bending: {
		upper = calcBending(ver, cons, consIndex);
		int4 verIndexList = cons.bending.ver[consIndex];
		for (int i = 0; i < 4; i++) {
			float3 gradient = make_float3(0, 0, 0);
			calcGradient(ver, cons, get(verIndexList, i), consIndex, tstep, type, gradient);
			float norm = norm2(gradient);
			lower += ver.invM[get(verIndexList, i)] * norm;
		}
		break;
	}
	}
	if (fabsf(lower) <= 1e-8f || !isfinite(lower)) {
		return 0.0f;
	}
	if (fabsf(upper) <= 1e-8f || !isfinite(upper)) {
		return 0.0f;
	}
	result = upper / lower;
	return result;
}
__device__ void GSiteration(VertexDevice ver, ConstraintDevice cons, int verIndex, float tstep, int iterationCount) {
	// GS 스타일의 즉시 업데이트는 제약사항 단위의 것을 의미하는 것이지 
	// 한 제약사항 내에서 각 정점마다 즉시 업데이트를 하면 안됨.

	int consIndex = 0;
	for (int count = 0; count < iterationCount; count++) {
		for (int i = 0; i < cons.stretch.n; i++) {
			calcDeltaP(ver, cons, verIndex, consIndex, Stretch, tstep);
			consIndex++;
		}
		consIndex = 0;
		for (int i = 0; i < cons.bending.n; i++) {
			calcDeltaP(ver, cons, verIndex, consIndex, Bending, tstep);
			consIndex++;
		}
	}
}
__device__ void updateVertices(VertexDevice ver, int verIndex, int tstep) {
	float3& p = ver.p[verIndex];
	float3& pos = ver.pos[verIndex];
	for (int i = 0; i < 3; i++) {
		float item = get(p, i) - get(pos, i);
		item /= tstep;
		set(ver.v[verIndex], i, item);
		set(pos, i, get(p, i));
	}
}

__global__ void applyForceKernel(VertexDevice ver, float3* forces, int forceCount, float tstep) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIndex >= ver.N) return;
	if (ver.invM[verIndex] == 0.0f) return;

	float3 totalForce = make_float3(0, 0, 0);
	for (int i = 0; i < forceCount; i++) {
		totalForce = add(totalForce, forces[i]);
	}
	float invM = ver.invM[verIndex];
	float3& curV = ver.v[verIndex];
	float3 accel = add(mul(totalForce, invM), make_float3(0,-9.8,0));
	curV = add(curV, mul(accel, tstep));
}

__global__ void computeDampingKernel(VertexDevice ver, DampingDevice damp) {
	if (blockIdx.x != 0 || threadIdx.x != 0) return;
	*damp.poscm = calcPoscm(ver);
	*damp.vcm = calcVcm(ver);
	*damp.omega = calcOmega(ver, *damp.poscm);
}

__global__ void applyDampingKernel(VertexDevice ver, DampingDevice damp, float k_damping) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIndex >= ver.N) return;
	if (ver.invM[verIndex] == 0.0f) return;

	float3 r = sub(ver.pos[verIndex], *damp.poscm);
	float3 omegaCrossR = cross(*damp.omega, r);

	float3 target = add(*damp.vcm, omegaCrossR);
	float3 deltaV = sub(target, ver.v[verIndex]);

	ver.v[verIndex] = add(ver.v[verIndex], mul(deltaV, k_damping));
}

__global__ void estimatePKernel(VertexDevice ver, float tstep) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIndex >= ver.N) return;
	ver.p[verIndex] = add(ver.pos[verIndex], mul(ver.v[verIndex], tstep));
}

__global__ void updateVerticesKernel(VertexDevice ver, float tstep) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIndex >= ver.N) return;
	if (ver.invM[verIndex] == 0.0f) return;

	float3 delta = sub(ver.p[verIndex], ver.pos[verIndex]);
	ver.v[verIndex] = mul(delta, 1.0f / tstep);
	ver.pos[verIndex] = ver.p[verIndex];
}

__global__ void solveConstraintsKernel(VertexDevice ver, ConstraintDevice cons, float tstep, int iterationCount) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIndex >= ver.N) return;
	if (ver.invM[verIndex] == 0.0f) return;

	int begin = ver.constraintOffset[verIndex];
	int end = ver.constraintOffset[verIndex + 1];

	for (int i = begin; i < end; i++) {
		unsigned int packed = ver.constraintsArray[i];
		Type type = unpackType(packed);
		int consIndex = (int)unpackIndex(packed);

		projectionFunction(ver, cons, verIndex, consIndex, type, tstep, iterationCount);
	}
}
__device__ float calcConstraintWeight(ConstraintDevice cons, int consIndex, Type type, int ns) {
	if (ns <= 0) return 0;
	switch (type) {
	case Stretch:
		return 1.0f - powf(1.0f - cons.stretch.k[consIndex], 1.0f / ns);
	case Bending:
		return 1.0f - powf(1.0f - cons.bending.k[consIndex], 1.0f / ns);
	}
}
__device__ void projectStretchConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount) {
	int2 ids = cons.stretch.ver[consIndex];
	float s = calcScale(ver, cons, ids.x, consIndex, Stretch, tstep);
	if (fabsf(s) < 1e-8f) return;

	float weight = calcConstraintWeight(cons, consIndex, Stretch, iterationCount);

	for (int i = 0; i < 2; i++) {
		int verIndex = get(ids, i);
		if (ver.invM[verIndex] == 0.0f) continue;

		float3 gradient = make_float3(0, 0, 0);
		calcGradient(ver, cons, verIndex, consIndex, tstep, Stretch, gradient);

		float3 delta = mul(gradient, -s * ver.invM[verIndex] * weight);
		ver.p[verIndex] = add(ver.p[verIndex], delta);
	}
}
__device__ void projectBendingConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount) {
	int4 ids = cons.bending.ver[consIndex];
	float s = calcScale(ver, cons, ids.x, consIndex, Bending, tstep);

	if (fabsf(s) < 1e-8f) return;

	float weight = calcConstraintWeight(cons, consIndex, Bending, iterationCount);
	for (int i = 0; i < 4; i++) {
		int verIndex = get(ids, i);
		if (ver.invM[verIndex] == 0.0f) continue;

		float3 gradient = make_float3(0, 0, 0);
		calcGradient(ver, cons, verIndex, consIndex, tstep, Bending, gradient);

		float3 delta = mul(gradient, -s * ver.invM[verIndex] * weight);
		ver.p[verIndex] = add(ver.p[verIndex], delta);
	}

}
__global__ void solveStretchColorKernel(VertexDevice ver, ConstraintDevice cons, const int* constraintIds, int count, float tstep, int iterationCount) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count) return;

	int consIndex = constraintIds[idx];
	projectStretchConstraint(ver, cons, consIndex, tstep, iterationCount);
}
__global__ void solveBendingColorKernel(VertexDevice ver, ConstraintDevice cons, const int* constraintIds, int count, float tstep, int iterationCount) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count) return;

	int consIndex = constraintIds[idx];
	projectBendingConstraint(ver, cons, consIndex, tstep, iterationCount);
}

__global__ void projectCollisionConstraint(VertexDevice ver, ConstraintDevice cons, int count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) return;

	int verIndex = cons.collision.ver[i];
	float3 p = ver.p[verIndex];
	float3 q = cons.collision.q[i];
	float3 n = normalize(cons.collision.normal[i]);
	float thickness = cons.collision.thickness[i];
	float C = dot(sub(p, q), n)- thickness;
	ver.p[verIndex] = add(p, mul(n, -C));
}
void solve(VertexDevice ver, ConstraintDevice cons, DampingDevice damp, std::vector<float*>& vertexSet, std::vector<unsigned int*>& indexSet, std::vector<int>& indexSetN, float3* forces, float k_damping, float tstep, float currentTime, int iterationCount, int forceCount, int n, std::vector<int> stretchColorOffset, std::vector<int> bendingColorOffset) {
	computeDampingKernel<<<1, 1>>> (ver, damp);
	checkCudaKernel("computeDampingKernel launch failed");
	int threads = 256;
	int blocks = (ver.N + threads - 1) / threads;

	applyForceKernel<<<blocks, threads>>> (ver, forces, forceCount, tstep);
	checkCudaKernel("applyForceKernel launch failed");
	applyDampingKernel<<<blocks, threads>>> (ver, damp, k_damping);
	checkCudaKernel("applyDampingKernel launch failed");
	estimatePKernel<<<blocks, threads>>> (ver, tstep);
	checkCudaKernel("estimatePKernel launch failed");

	int zero = 0;
	cudaMemcpy(cons.collision.n, &zero, sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 0; i < vertexSet.size(); i++) {
		int triangleCount = indexSetN[i]/3;
		detectCollisionKernel <<<blocks, threads >>> (ver, cons, vertexSet[i], indexSet[i], triangleCount);
		checkCudaKernel("detectCollisionKernel launch failed");
	}
	int collisionCount = 0;
	cudaMemcpy(&collisionCount, cons.collision.n, sizeof(int), cudaMemcpyDeviceToHost);
	//std::cout << "collisionCount = " << collisionCount << std::endl;

	for (int iter = 0; iter < iterationCount; iter++) {		
		for (int color = 0; color < cons.stretch.color.colorCount; color++) {
			int start = stretchColorOffset[color];
			int end = stretchColorOffset[color + 1];
			int count = end - start;
			if (count <= 0)continue;

			int colorBlocks = (count + threads - 1) / threads;
			solveStretchColorKernel<<<colorBlocks, threads >>> (ver, cons, cons.stretch.color.constraintIds + start, count, tstep, iterationCount);
			checkCudaKernel("solveStretchColorKernel launch failed");
		}
		for (int color = 0; color < cons.bending.color.colorCount; color++) {
			int start = bendingColorOffset[color];
			int end = bendingColorOffset[color + 1];
			int count = end - start;
			if (count <= 0)continue;

			int colorBlocks = (count + threads - 1) / threads;
			solveBendingColorKernel<<<colorBlocks, threads >>> (ver, cons, cons.bending.color.constraintIds + start, count, tstep, iterationCount);
			checkCudaKernel("solveBendingColorKernel launch failed");
		}
		if (collisionCount > 0) {
			int collisionBlocks = (collisionCount + threads - 1) / threads;
			projectCollisionConstraint << <collisionBlocks, threads >> > (ver, cons, collisionCount);
			checkCudaKernel("projectCollisionConstraint launch failed");
		}
		
		clearVectorKernel<<<blocks, threads>>>(ver.dp, ver.N);
		checkCudaKernel("clearVectorKernel launch failed");
		checkCudaKernel("applyDeltaToPredictedKernel launch failed");
		clearVectorKernel << <blocks, threads >> > (ver.dp, ver.N);
		checkCudaKernel("clearVectorKernel launch failed");
	}
	updateVerticesKernel<<<blocks, threads>>>(ver, tstep);
	checkCudaKernel("updateVerticesKernel launch failed");

}