#include "XPBDSolver.cuh"
#include "CommonSolver.cuh"
#include "CudaUtils.cuh"

#include <math_constants.h>
#include <device_launch_parameters.h>

namespace xpbd {

	__global__ void sumForceKernel(const float3* forces, int forceCount, float3* totalForce) {
		if (blockIdx.x != 0 || threadIdx.x != 0) {
			return;
		}

		float3 result = make_float3(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < forceCount; i++) {
			result = add(result, forces[i]);
		}

		*totalForce = result;
	}
	__global__ void predictPositionKernel(VertexDevice ver, const float3* totalForce, float tstep) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= ver.N) {
			return;
		}

		if (ver.invM[idx] == 0.0f) {
			return;
		}
		float3 force = *totalForce;
		float3 acceleration = mul(force, ver.invM[idx]);

		ver.p[idx] = add(add(ver.pos[idx], mul(ver.v[idx], tstep)), mul(force, ver.invM[idx] * tstep * tstep));
	}
	__global__ void projectMouseDragConstraintKernel(VertexDevice ver, int vertexIndex, float3 target, float compliance, float tstep, float3* lambda) {
		if (blockIdx.x != 0 || threadIdx.x != 0) {
			return;
		}
		if (vertexIndex < 0 || vertexIndex >= ver.N) {
			return;
		}
		if (lambda == nullptr || !isfinite(compliance) || compliance < 0.0f || !isfinite(tstep) || tstep <= 0.0f) {
			return;
		}
		float invM = ver.invM[vertexIndex];
		if (invM <= 0.0f) {
			return;
		}
		float scaledCompliance = calcScaledCompliance(compliance, tstep);
		float denominator = invM + scaledCompliance;
		if (!isfinite(denominator) || denominator <= 1.0e-8f) {
			return;
		}
		float3 p = ver.p[vertexIndex];
		float3 C = sub(p, target);
		float3 oldLambda = *lambda;

		float3 numerator = sub(mul(C, -1.0f), mul(oldLambda, scaledCompliance));
		float3 deltaLambda = mul(numerator, 1.0f / denominator);

		if (!isfinite(deltaLambda.x) || !isfinite(deltaLambda.y) || !isfinite(deltaLambda.z)) {
			return;
		}
		*lambda = add(oldLambda, deltaLambda);
		ver.p[vertexIndex] = add(p, mul(deltaLambda, invM));
	}
	__device__ float calcScaledCompliance(float compliance, float tstep) {
		if (!isfinite(compliance) || compliance < 0.0f || !isfinite(tstep) || tstep <= 0.0f) {
			return CUDART_INF_F;
		}
		return compliance / ((tstep) * (tstep));
	}
	__device__ float calcDampingGamma(float compliance, float dampingStiff, float tstep) {
		if (!isfinite(compliance) || compliance <= 1e-8f || !isfinite(dampingStiff) || dampingStiff <= 0.0f || !isfinite(tstep) || tstep <= 0.0f) {
			return 0.0f;
		}
		return compliance * dampingStiff / tstep;
	}
	__device__ void projectStretchConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float dampingStiff, float tstep, int iterationCount, float& lambda) {
		float compliance = cons.stretch.k[consIndex];

		float scaledCompliance = calcScaledCompliance(compliance, tstep);

		float gamma = calcDampingGamma(compliance, dampingStiff, tstep);

		int2 ids = cons.stretch.ver[consIndex];
		float3 p0 = ver.p[ids.x];
		float3 p1 = ver.p[ids.y];

		float3 diff = sub(p0, p1);
		float currentLength = length(diff);
		float restLength = cons.stretch.l0[consIndex];

		float C = currentLength - restLength;
		if (currentLength <= 1e-8f) {
			return;
		}
		float3 gradient0 = mul(diff, 1.0f / currentLength);
		float3 gradient1 = mul(gradient0, -1.0f);
		
		float effectiveMass = ver.invM[ids.x] * dot(gradient0, gradient0) + ver.invM[ids.y] * dot(gradient1, gradient1);
		if (!isfinite(effectiveMass) || effectiveMass <= 1e-8f) {
			return;
		}
		float constraintDisplacement = dot(gradient0, sub(p0, ver.pos[ids.x])) + dot(gradient1, sub(p1, ver.pos[ids.y]));

		float upper = -C - scaledCompliance * lambda - gamma * constraintDisplacement;
		float lower = (1.0f + gamma) * effectiveMass + scaledCompliance;
		if (!isfinite(lower) || lower <= 1e-8f) {
			return;
		}
		float deltaLambda = upper / lower;
		float3 deltaPos0 = mul(gradient0, ver.invM[ids.x] * deltaLambda);
		float3 deltaPos1 = mul(gradient1, ver.invM[ids.y] * deltaLambda);

		lambda += deltaLambda;
		p0 = add(deltaPos0, p0);
		p1 = add(deltaPos1, p1);

		ver.p[ids.x] = p0;
		ver.p[ids.y] = p1;
	}
	__device__ void projectBendingConstraint(VertexDevice ver, ConstraintDevice cons, int consIndex, float dampingStiff, float tstep, int iterationCount, float& lambda) {
		float compliance = cons.bending.k[consIndex];
		if (!isfinite(compliance)) {
			return;
		}
		float scaledCompliance = calcScaledCompliance(compliance, tstep);
		float gamma = calcDampingGamma(compliance, dampingStiff, tstep);

		int4 ids = cons.bending.ver[consIndex];

		float3 p0 = ver.p[ids.x];
		float3 p1 = ver.p[ids.y];
		float3 p2 = ver.p[ids.z];
		float3 p3 = ver.p[ids.w];

		float3 e = sub(p1, p0);
		float3 a = sub(p2, p0);
		float3 b = sub(p3, p0);
		float eLength = length(e);
		if (eLength <= 1e-8f) {
			return;
		}
		float3 ehat = mul(e, 1.0f / eLength);

		float3 n0Upper = cross(e, a);
		float n0Lower = length(n0Upper);
		if (n0Lower <= 1e-8) {
			return;
		}
		float3 n0 = mul(n0Upper, 1.0f / n0Lower);

		float3 n1Upper = cross(e, b);
		float n1Lower = length(n1Upper);
		if (n1Lower <= 1e-8) {
			return;
		}
		float3 n1 = mul(n1Upper, 1.0f / n1Lower);

		float d = dot(n0, n1);
		d = fminf(1.0f, fmaxf(-1.0f, d));

		float3 angleN0 = mul(n0, -1.0f);
		float3 angleN1 = n1;
		float currentAngle = atan2f(dot(ehat, cross(angleN1, angleN0)), dot(angleN0, angleN1));
		//float currentAngle = acos(d);
		float restAngle = cons.bending.phi0[consIndex];

		float C = currentAngle - restAngle;
		// 각도가 179-(-179)=358로 계산되지 않고 2로 계산되도록 보정
		if (C > CUDART_PI_F) {
			C -= 2.0f * CUDART_PI_F;
		}
		else if (C < -CUDART_PI_F) {
			C += 2.0f * CUDART_PI_F;
		}

		if (!isfinite(C) || fabsf(C) <= 1e-8f) {
			return;
		}
		float eLengthSquared = eLength * eLength;
		float3 gradient2 = mul(n0Upper, eLength / (n0Lower * n0Lower));
		float3 gradient3 = mul(n1Upper, -eLength / (n1Lower * n1Lower));
		float3 gradient0 = add(mul(gradient2, dot(sub(p2, p1), e) / eLengthSquared), mul(gradient3, dot(sub(p3, p1), e) / eLengthSquared));
		float3 gradient1 = add(mul(gradient2, -dot(sub(p2, p0), e) / eLengthSquared), mul(gradient3, -dot(sub(p3, p0), e) / eLengthSquared));

		float effectiveMass = ver.invM[ids.x] * dot(gradient0, gradient0) + ver.invM[ids.y] * dot(gradient1, gradient1) + ver.invM[ids.z] * dot(gradient2, gradient2) + ver.invM[ids.w] * dot(gradient3, gradient3);
		if (!isfinite(effectiveMass) || effectiveMass <= 1e-8f) {
			return;
		}
		float constraintDisplacement = dot(gradient0, sub(p0, ver.pos[ids.x])) + dot(gradient1, sub(p1, ver.pos[ids.y])) + dot(gradient2, sub(p2, ver.pos[ids.z])) + dot(gradient3, sub(p3, ver.pos[ids.w]));

		float upper = -C - scaledCompliance * lambda - gamma * constraintDisplacement;
		float lower = (1.0f + gamma) * effectiveMass + scaledCompliance;
		if (!isfinite(lower) || lower <= 1e-8f) {
			return;
		}
		float deltaLambda = upper / lower;
		float3 deltaPos0 = mul(gradient0, ver.invM[ids.x] * deltaLambda);
		float3 deltaPos1 = mul(gradient1, ver.invM[ids.y] * deltaLambda);
		float3 deltaPos2 = mul(gradient2, ver.invM[ids.z] * deltaLambda);
		float3 deltaPos3 = mul(gradient3, ver.invM[ids.w] * deltaLambda);

		lambda += deltaLambda;
		p0 = add(deltaPos0, p0);
		p1 = add(deltaPos1, p1);
		p2 = add(deltaPos2, p2);
		p3 = add(deltaPos3, p3);

		ver.p[ids.x] = p0;
		ver.p[ids.y] = p1;
		ver.p[ids.z] = p2;
		ver.p[ids.w] = p3;
	}
	__device__ float calcUnilateralDeltaLambda(float C, float stiff, float tstep, float denom, float& lambda) {
		if (tstep <= 0.0f || denom < 0.0f) {
			return 0.0f;
		}
		stiff = fmaxf(stiff, 0.0f);
		float scaledCompliance = stiff / (tstep * tstep);
		float xpbdDenom = denom + scaledCompliance;

		if (xpbdDenom < 1e-8f || !isfinite(xpbdDenom)) {
			return 0.0f;
		}
		float deltaLambda = (-C - scaledCompliance * lambda) / xpbdDenom;
		float newLambda = fmaxf(0.0f, lambda + deltaLambda);

		if (!isfinite(newLambda)) {
			return 0.0f;
		}
		deltaLambda = newLambda - lambda;
		lambda = newLambda;

		return deltaLambda;
	}
	__global__ void projectCollisionConstraint(VertexDevice ver, ConstraintDevice cons, float tstep) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= *(cons.collision.n)) {
			return;
		}

		int collisionCount = min(*(cons.collision.n), cons.collision.capacity);

		if (i >= collisionCount) {
			return;
		}

		int verIndex = cons.collision.ver[i];
		if (verIndex < 0 || verIndex >= ver.N) {
			return;
		}
		float invM = ver.invM[verIndex];
		if (invM <= 0.0f) {
			return;
		}
		float3 p = ver.p[verIndex];
		float3 q = cons.collision.q[i];
		float3 rawNormal = cons.collision.normal[i];

		float normalLengthSquared = dot(rawNormal, rawNormal);
		if (normalLengthSquared < 1e-8f || !isfinite(normalLengthSquared)) {
			return;
		}
		float3 n = mul(rawNormal, rsqrtf(normalLengthSquared));

		float thickness = cons.collision.thickness[i];
		float compliance = cons.collision.compliance[i];

		float C = dot(sub(p, q), n) - thickness;
		float& lambda = cons.collision.lambda[i];
		
		if (C >= 0.0f && lambda <= 0.0f) {
			return;
		}
		float deltaLambda = calcUnilateralDeltaLambda(C, compliance, tstep, invM, lambda);
		if (fabsf(deltaLambda) < 1e-8f || !isfinite(deltaLambda)) {
			return;
		}
		float3 corr = mul(n, invM * deltaLambda);
		if (!isfinite(corr.x) || !isfinite(corr.y) || !isfinite(corr.z)) {
			return;
		}
		atomicAddFloat3(ver.dp, verIndex, corr);
		atomicAdd(&(ver.dpCount[verIndex]), 1);
	}

	__global__ void projectSelfCollisionConstraintKernel(VertexDevice ver, ConstraintDevice cons, float tstep) {
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

		float3 rawNormal = cons.selfCollision.normal[i];
		float normalLengthSquared = dot(rawNormal, rawNormal);

		if (normalLengthSquared < 1e-12f || !isfinite(normalLengthSquared)) {
			return;
		}
		float3 n = mul(rawNormal, rsqrtf(normalLengthSquared));
		float thickness = cons.selfCollision.thickness[i];
		float compliance = cons.selfCollision.compliance[i];

		float3 bary = barycentric(p0, p1, p2, cons.selfCollision.q[i]);

		if (!isfinite(bary.x) || !isfinite(bary.y) || !isfinite(bary.z)) {
			return;
		}
		if (bary.x < -1e-5f || bary.y < -1e-5f || bary.z < -1e-5f) {
			return;
		}
		float b0 = bary.x;
		float b1 = bary.y;
		float b2 = bary.z;

		float3 triPoint = add(add(mul(p0, b0), mul(p1, b1)), mul(p2, b2));
		float C = dot(sub(q, triPoint), n) - thickness;
		float& lambda = cons.selfCollision.lambda[i];

		if (C >= 0.0f && lambda <= 0.0f) {
			return;
		}

		float wq = ver.invM[qIndex];
		float w0 = ver.invM[i0];
		float w1 = ver.invM[i1];
		float w2 = ver.invM[i2];

		if (wq <= 0.0f && w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f) {
			return;
		}

		float denom = wq + w0 * b0 * b0 + w1 * b1 * b1 + w2 * b2 * b2;
		if (denom < 1e-8f || !isfinite(denom)) {
			return;
		}
		float deltaLambda = calcUnilateralDeltaLambda(C, compliance, tstep, denom, lambda);
		if (fabsf(deltaLambda) < 1e-8f || !isfinite(deltaLambda)) {
			return;
		}

		float3 dq = mul(n, wq * deltaLambda);
		float3 d0 = mul(n, -w0 * b0 * deltaLambda);
		float3 d1 = mul(n, -w1 * b1 * deltaLambda);
		float3 d2 = mul(n, -w2 * b2 * deltaLambda);

		if (wq > 0.0f && isfinite(dq.x) && isfinite(dq.y) && isfinite(dq.z)) {
			atomicAddFloat3(ver.dp, qIndex, dq);
			atomicAdd(&(ver.dpCount[qIndex]), 1);
		}
		if (w0 > 0.0f && isfinite(d0.x) && isfinite(d0.y) && isfinite(d0.z)) {
			atomicAddFloat3(ver.dp, i0, d0);
			atomicAdd(&(ver.dpCount[i0]), 1);
		}
		if (w1 > 0.0f && isfinite(d1.x) && isfinite(d1.y) && isfinite(d1.z)) {
			atomicAddFloat3(ver.dp, i1, d1);
			atomicAdd(&(ver.dpCount[i1]), 1);
		}
		if (w2 > 0.0f && isfinite(d2.x) && isfinite(d2.y) && isfinite(d2.z)) {
			atomicAddFloat3(ver.dp, i2, d2);
			atomicAdd(&(ver.dpCount[i2]), 1);
		}
	}
	__global__ void resetCollisionLambdaKernel(ConstraintDevice cons) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int collisionCount = min(*(cons.collision.n), cons.collision.capacity);

		if (i >= collisionCount) {
			return;
		}

		cons.collision.lambda[i] = 0.0f;
	}
	__global__ void resetSelfCollisionLambdaKernel(ConstraintDevice cons) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int selfCollisionCount = min(*(cons.selfCollision.n), cons.selfCollision.capacity);

		if (i >= selfCollisionCount) {
			return;
		}

		cons.selfCollision.lambda[i] = 0.0f;
	}
	void solve(VertexDevice ver, ConstraintDevice cons, DampingDevice damp, CudaConstraintGraph& constraintIterationGraph, std::vector<float*>& vertexSet, std::vector<float*>& prevVertexSet, std::vector<unsigned int*>& indexSet, std::vector<int>& indexSetN, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, unsigned int* d_gridHashes, float* d_totalMass, float3* d_totalForce, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float cellSize, float selfThickness, float selfStiffness, int gridCapacity, float3* forces, float stretchDamping, float bendingDamping, float tstep, float currentTime, int iterationCount, int forceCount, int n, std::vector<int>& stretchColorOffset, std::vector<int>& bendingColorOffset, const float friction, const float restitution, bool mouseDragActive, int mouseDragVertex, float3 mouseDragTarget, float mouseDragCompliance, float3* mouseDragLambda) {
		int threads = 256;
		int blocks = (ver.N + threads - 1) / threads;

		sumForceKernel << <1, 1 >> > (forces, forceCount, d_totalForce);
		checkCudaKernel("sumForceKernel launch failed");

		predictPositionKernel << <blocks, threads >> > (ver, d_totalForce, tstep);
		checkCudaKernel("predictPositionKernel launch failed");

		if (mouseDragActive && mouseDragLambda != nullptr) {
			checkCuda(cudaMemset(mouseDragLambda, 0, sizeof(float3)), "cudaMemset mouseDragLambda failed");
		}
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
		
		if (!constraintIterationGraph.isInitialized()) {
			constraintIterationGraph.build([&](cudaStream_t stream) {
				for (int color = 0; color < cons.stretch.color.colorCount; color++) {
					int start = stretchColorOffset[color];
					int end = stretchColorOffset[color + 1];
					int count = end - start;

					if (count <= 0) continue;

					int colorBlocks = (count + threads - 1) / threads;

					solveStretchColorKernel << <colorBlocks, threads, 0, stream >> > (ver, cons, cons.stretch.color.constraintIds + start, count, tstep, iterationCount, xpbd::StretchProjector{ cons.stretch.lambda, stretchDamping });
				}

				for (int color = 0; color < cons.bending.color.colorCount; color++) {
					int start = bendingColorOffset[color];
					int end = bendingColorOffset[color + 1];
					int count = end - start;

					if (count <= 0) continue;

					int colorBlocks = (count + threads - 1) / threads;

					solveBendingColorKernel << <colorBlocks, threads, 0, stream >> > (ver, cons, cons.bending.color.constraintIds + start, count, tstep, iterationCount, xpbd::BendingProjector{ cons.bending.lambda, bendingDamping });
				}
				checkCuda(cudaMemsetAsync(cons.collision.n, 0, sizeof(int), stream), "cudaMemsetAsync collision.n failed");
				});
		}
		if (cons.stretch.n > 0) {
			checkCuda(cudaMemset(cons.stretch.lambda, 0, sizeof(float) * cons.stretch.n), "cudaMemset stretch.lambda failed");
		}

		if (cons.bending.n > 0) {
			checkCuda(cudaMemset(cons.bending.lambda, 0, sizeof(float) * cons.bending.n), "cudaMemset bending.lambda failed");
		}
		for (int iter = 0; iter < iterationCount; iter++) {
			if (mouseDragActive) {
				projectMouseDragConstraintKernel << <1, 1 >> > (ver, mouseDragVertex, mouseDragTarget, mouseDragCompliance, tstep, mouseDragLambda);
				checkCudaKernel("projectMouseDragConstraintKernel pre-solve failed");
			}
			constraintIterationGraph.launch();
			if (mouseDragActive) {
				projectMouseDragConstraintKernel << <1, 1 >> > (ver, mouseDragVertex, mouseDragTarget, mouseDragCompliance, tstep, mouseDragLambda);
				checkCudaKernel("projectMouseDragConstraintKernel post-solve failed");
			}
			bool runCCD = (iter == 0) || ((iter % CCD_INTERVAL) == 0);
			for (int i = 0; i < vertexSet.size(); i++) {
				int triangleCount = indexSetN[i] / 3;
				float* prevColliderVertices = vertexSet[i];
				if (i < prevVertexSet.size() && prevVertexSet[i] != nullptr) {
					prevColliderVertices = prevVertexSet[i];
				}
				if (runCCD) {
					detectContinuousCollisionKernel << <blocks, threads >> > (ver, cons, vertexSet[i], prevColliderVertices, indexSet[i], triangleCount, tstep);
					checkCudaKernel("detectContinuousCollisionKernel CCD launch failed");
				}
				detectStaticCollisionKernel << <blocks, threads >> > (ver, cons, vertexSet[i], prevColliderVertices, indexSet[i], triangleCount, tstep);
				checkCudaKernel("detectStaticCollisionKernel launch failed");
				clearVectorKernel << <blocks, threads >> > (ver.dp, ver.N);
				checkCudaKernel("clearVectorKernel ver.dp launch failed");
				clearIntKernel << <blocks, threads >> > (ver.dpCount, ver.N);
				checkCudaKernel("clearIntKernel ver.dpCount launch failed");

				projectCollisionConstraint << <staticCollBlocks, threads >> > (ver, cons, tstep);
				projectSelfCollisionConstraintKernel << <selfCollBlocks, threads >> > (ver, cons, tstep);

				applyAverageDeltaToPredictedKernel << <blocks, threads >> > (ver);
				checkCudaKernel("applyAverageDeltaToPredictedKernel launch failed");
			}
		}
		updateVerticesKernel << <blocks, threads >> > (ver, tstep);
		checkCudaKernel("updateVerticesKernel launch failed");
		velocityUpdateKernel << <blocks, threads >> > (ver, cons, friction, restitution);
		checkCudaKernel("velocityUpdateKernel launch failed");
	}
}