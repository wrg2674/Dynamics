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
	if (fabsf(lower) <= 1e-6f || !isfinite(lower)) {
		return 0.0f;
	}
	if (fabsf(upper) <= 1e-6f || !isfinite(upper)) {
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
__device__ void updateVertices(VertexDevice ver, int verIndex, float tstep) {
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
__global__ void applyAverageDeltaToPredictedKernel(VertexDevice ver){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= ver.N) {
		return;
	}

	if (ver.invM[i] == 0.0f) {
		return;
	}

	int count = ver.dpCount[i];

	if (count <= 0) {
		return;
	}

	float invCount = 1.0f / (float)count;
	float3 avgDelta = mul(ver.dp[i], invCount);

	ver.p[i] = add(ver.p[i], avgDelta);

}
__global__ void initDampingVariablesKernel(DampingDevice damp, float* d_totalMass) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*damp.poscm = make_float3(0, 0, 0);
		*damp.vcm = make_float3(0, 0, 0);
		*damp.omega = make_float3(0, 0, 0);
		*d_totalMass = 0.0f;
	}
}
__global__ void computeDampingKernel(VertexDevice ver, DampingDevice damp, float* d_totalMass) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	if (i >= ver.N || ver.invM[i] == 0.0f) return;

	float m = 1.0f / ver.invM[i];
	atomicAddFloat3(damp.poscm, mul(ver.pos[i], m));
	atomicAddFloat3(damp.vcm, mul(ver.v[i], m));
	atomicAdd(d_totalMass, m);
}

__global__ void finalizeDampingKernel(DampingDevice damp, float* d_totalMass) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		float M = *d_totalMass;
		if (M > 0) {
			*damp.poscm = mul(*damp.poscm, 1.0f / M);
			*damp.vcm = mul(*damp.vcm, 1.0f / M);
		}
	}
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
	if (ns <= 0) return 0.0f;
	switch (type) {
	case Stretch:
		return 1.0f - powf(1.0f - cons.stretch.k[consIndex], 1.0f / ns);
	case Bending:
		return 1.0f - powf(1.0f - cons.bending.k[consIndex], 1.0f / ns);
	default:
		return 0.0f;
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
	if (i >= *(cons.collision.n)) return;

	int collisionCount = min(*(cons.collision.n), cons.collision.capacity);
	if (i >= collisionCount) return;

	int verIndex = cons.collision.ver[i];
	if (ver.invM[verIndex] == 0.0f) {
		return;
	}
	float3 p = ver.p[verIndex];
	float3 q = cons.collision.q[i];
	float3 n = normalize(cons.collision.normal[i]);
	float thickness = cons.collision.thickness[i];
	float stiffness = clamp(cons.collision.k[i], 0.0f, 1.0f);

	float C = dot(sub(p, q), n)- thickness;
	if (C >= 0.0f) {
		return;
	}
	float3 corr = mul(n, -C * stiffness);
	if (!isfinite(corr.x) || !isfinite(corr.y) || !isfinite(corr.z)) {
		return;
	}
	atomicAddFloat3(ver.dp, verIndex, corr);
	atomicAdd(&(ver.dpCount[verIndex]), 1);
}
__global__ void projectSelfCollisionConstraintKernel(VertexDevice ver, ConstraintDevice cons) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int selfCollisionCount = min(*(cons.selfCollision.n), cons.selfCollision.capacity);

	if (i >= selfCollisionCount) {
		return;
	}
	int qIndex = cons.selfCollision.ver[i];
	if (qIndex < 0 || qIndex >= ver.N) {
		return;
	}

	if (ver.invM[qIndex] == 0.0f) {
		return;
	}
	int3 tri = cons.selfCollision.tri[i];

	int i0 = tri.x;
	int i1 = tri.y;
	int i2 = tri.z;

	if (i0 < 0 || i0 >= ver.N) {
		return;
	}
	if (i1 < 0 || i1 >= ver.N) {
		return;
	}
	if (i2 < 0 || i2 >= ver.N) {
		return;
	}
	if (qIndex == i0 || qIndex == i1 || qIndex == i2) {
		return;
	}
	float3 q = ver.p[qIndex];
	float3 p0 = ver.p[i0];
	float3 p1 = ver.p[i1];
	float3 p2 = ver.p[i2];

	float3 n = normalize(cons.selfCollision.normal[i]);
	float thickness = cons.selfCollision.thickness[i];
	float stiffness = clamp(cons.selfCollision.k[i], 0.0f, 1.0f);

	float3 bary = barycentric(p0, p1, p2, cons.selfCollision.q[i]);

	float b0 = clamp(bary.x, 0.0f, 1.0f);
	float b1 = clamp(bary.y, 0.0f, 1.0f);
	float b2 = clamp(bary.z, 0.0f, 1.0f);

	float bSum = b0 + b1 + b2;
	if (bSum < 1e-8f || !isfinite(bSum)) {
		return;
	}
	b0 /= bSum;
	b1 /= bSum;
	b2 /= bSum;

	float3 triPoint = add(add(mul(p0, b0), mul(p1, b1)), mul(p2, b2));
	float C = dot(sub(q, triPoint), n) - thickness;

	if (C >= 0.0f) {
		return;
	}
	float wq = ver.invM[qIndex];
	float w0 = ver.invM[i0];
	float w1 = ver.invM[i1];
	float w2 = ver.invM[i2];

	float denom = wq + w0 * b0 * b0 + w1 * b1 * b1 + w2 * b2 * b2;
	if (denom < 1e-8f || !isfinite(denom)) {
		return;
	}
	float s = C / denom;

	float3 dq = mul(n, -s * wq * stiffness);
	float3 d0 = mul(n, s * w0 * b0 * stiffness);
	float3 d1 = mul(n, s * w1 * b1 * stiffness);
	float3 d2 = mul(n, s * w2 * b2 * stiffness);

	if (wq > 0.0f) {
		atomicAddFloat3(ver.dp, qIndex, dq);
		atomicAdd(&(ver.dpCount[qIndex]), 1);
	}
	if (w0 > 0.0f) {
		atomicAddFloat3(ver.dp, i0, d0);
		atomicAdd(&(ver.dpCount[i0]), 1);
	}
	if (w1 > 0.0f) {
		atomicAddFloat3(ver.dp, i1, d1);
		atomicAdd(&(ver.dpCount[i1]), 1);
	}
	if (w2 > 0.0f) {
		atomicAddFloat3(ver.dp, i2, d2);
		atomicAdd(&(ver.dpCount[i2]), 1);
	}
}

__global__ void velocityUpdateKernel(VertexDevice ver, ConstraintDevice cons, float friction, float restitution) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIndex >= ver.N) return;

	if (ver.invM[verIndex] == 0.0f) {
		ver.v[verIndex] = make_float3(0.0f, 0.0f, 0.0f);
		return;
	}

	int collisionCount = min(*(cons.collision.n), cons.collision.capacity);
	float3 v = ver.v[verIndex];

	for (int i = 0; i < collisionCount; i++) {
		if (cons.collision.ver[i] != verIndex) {
			continue;
		}

		float3 n = normalize(cons.collision.normal[i]);
		float3 colliderV = cons.collision.colliderVelocity[i];

		float3 relV = sub(v, colliderV);
		float relVnMag = dot(relV, n);

		if (relVnMag >= 0.0f) {
			continue;
		}

		float3 relVn = mul(n, relVnMag);
		float3 relVt = sub(relV, relVn);

		float3 newRelVn = mul(n, -relVnMag * restitution);
		float3 newRelVt = mul(relVt, fmaxf(0.0f, 1.0f - friction));

		float3 newRelV = add(newRelVn, newRelVt);

		v = add(newRelV, colliderV);
	}

	ver.v[verIndex] = v;
}


void solve(VertexDevice ver, ConstraintDevice cons, DampingDevice damp, std::vector<float*>& vertexSet, std::vector<float*>& prevVertexSet, std::vector<unsigned int*>& indexSet, std::vector<int>& indexSetN, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, unsigned int* d_gridHashes, float* d_totalMass, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float cellSize, float selfThickness, float selfStiffness, int gridCapacity, float3* forces, float k_damping, float tstep, float currentTime, int iterationCount, int forceCount, int n, std::vector<int> stretchColorOffset, std::vector<int> bendingColorOffset, const float friction, const float restitution) {
	
	checkCudaKernel("computeDampingKernel launch failed");
	int threads = 256;
	int blocks = (ver.N + threads - 1) / threads;
	initDampingVariablesKernel << <1, 1 >> > (damp, d_totalMass);
	computeDampingKernel << <blocks, threads >> > (ver, damp, d_totalMass);
	finalizeDampingKernel << <1, 1 >> > (damp, d_totalMass);

	applyForceKernel<<<blocks, threads>>> (ver, forces, forceCount, tstep);
	checkCudaKernel("applyForceKernel launch failed");
	applyDampingKernel<<<blocks, threads>>> (ver, damp, k_damping);
	checkCudaKernel("applyDampingKernel launch failed");
	estimatePKernel<<<blocks, threads>>> (ver, tstep);
	checkCudaKernel("estimatePKernel launch failed");

	int staticCollBlocks = (cons.collision.capacity + threads - 1) / threads;
	int selfCollBlocks = (cons.selfCollision.capacity + threads - 1) / threads;
	updateSpatialHash(ver, cellSize, d_gridHashes, d_gridIndices, d_cellStart, d_cellEnd, gridCapacity);

	int zero = 0;
	cudaMemset(cons.selfCollision.n, 0, sizeof(int));
	checkCudaKernel("cudaMemset selfCollision.n failed");

	if (d_gridIndices != nullptr && d_cellStart != nullptr && d_cellEnd != nullptr && selfTris != nullptr && vertTriArray != nullptr && vertTriOffset != nullptr) {
		detectSelfCollisionKernel << <blocks, threads >> > (ver, cons, d_gridIndices, d_cellStart, d_cellEnd, selfTris, vertTriArray, vertTriOffset, cellSize, selfThickness, selfStiffness);
		checkCudaKernel("detectSelfCollisionKernel launch failed");
	}

	for (int iter = 0; iter < iterationCount; iter++) {		
		for (int color = 0; color < cons.stretch.color.colorCount; color++) {
			int start = stretchColorOffset[color];
			int end = stretchColorOffset[color + 1];
			int count = end - start;
			if (count <= 0) {
				continue;
			}
			int colorBlocks = (count + threads - 1) / threads;
			solveStretchColorKernel<<<colorBlocks, threads >>> (ver, cons, cons.stretch.color.constraintIds + start, count, tstep, iterationCount);
			checkCudaKernel("solveStretchColorKernel launch failed");
		}
		for (int color = 0; color < cons.bending.color.colorCount; color++) {
			int start = bendingColorOffset[color];
			int end = bendingColorOffset[color + 1];
			int count = end - start;
			if (count <= 0) {
				continue;
			}

			int colorBlocks = (count + threads - 1) / threads;
			solveBendingColorKernel<<<colorBlocks, threads >>> (ver, cons, cons.bending.color.constraintIds + start, count, tstep, iterationCount);
			checkCudaKernel("solveBendingColorKernel launch failed");
		}
		cudaMemset(cons.collision.n, 0, sizeof(int));
		checkCudaKernel("cudaMemset collision.n failed");

		bool runCCD = (iter == 0) || ((iter % CCD_INTERVAL) == 0);
		for (int i = 0; i < vertexSet.size(); i++) {
			int triangleCount = indexSetN[i] / 3;
			if (runCCD) {
				detectContinuousCollisionKernel << <blocks, threads >> > (ver, cons, vertexSet[i], indexSet[i], triangleCount);
				checkCudaKernel("detectContinuousCollisionKernel CCD launch failed");
			}
			detectStaticCollisionKernel << <blocks, threads >> > (ver, cons, vertexSet[i], indexSet[i], triangleCount);
			checkCudaKernel("detectStaticCollisionKernel launch failed");
			clearVectorKernel << <blocks, threads >> > (ver.dp, ver.N);
			checkCudaKernel("clearVectorKernel ver.dp launch failed");
			clearIntKernel << <blocks, threads >> > (ver.dpCount, ver.N);
			checkCudaKernel("clearIntKernel ver.dpCount launch failed");
			projectCollisionConstraint << <staticCollBlocks, threads >> > (ver, cons, 0);
			projectSelfCollisionConstraintKernel << <selfCollBlocks, threads >> > (ver, cons);
			
			
			applyAverageDeltaToPredictedKernel << <blocks, threads >> > (ver);
			checkCudaKernel("applyAverageDeltaToPredictedKernel launch failed");
		}
	}
	updateVerticesKernel<<<blocks, threads>>>(ver, tstep);
	checkCudaKernel("updateVerticesKernel launch failed");
	//velocityUpdateKernel << <blocks, threads >> > (ver, cons, friction, restitution);
	//checkCudaKernel("velocityUpdateKernel launch failed");

}