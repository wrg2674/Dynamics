#include "PBDSolver.cuh"
#include "CommonSolver.cuh"
#include "CudaUtils.cuh"

#include <device_launch_parameters.h>

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
	float3 accel = add(mul(totalForce, invM), make_float3(0, -9.8, 0));
	curV = add(curV, mul(accel, tstep));
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

	float C = calcBending(ver, cons, consIndex);
	if (fabsf(C) < 1e-8f || !isfinite(C)) {
		return;
	}

	float3 gradients[4];
	calcBendingGradient(ver, cons, consIndex, gradients);

	int verIds[4] = { ids.x,ids.y,ids.z,ids.w };
	float lower = 0.0f;

	for (int i = 0; i < 4; i++) {
		int verIndex = verIds[i];
		if (verIndex < 0 || verIndex >= ver.N) {
			return;
		}

		float w = ver.invM[verIndex];
		if (w == 0.0f) {
			continue;
		}

		float g2 = norm2(gradients[i]);
		if (!isfinite(g2)) {
			return;
		}
		lower += w * g2;
	}

	if (fabsf(lower) < 1e-8f || !isfinite(lower)) {
		return;
	}

	float s = C / lower;
	if (fabsf(s) < 1e-8f || !isfinite(s)) {
		return;
	}

	float weight = calcConstraintWeight(cons, consIndex, Bending, iterationCount);
	if (fabsf(weight) < 1e-8f || !isfinite(weight)) {
		return;
	}

	for (int i = 0; i < 4; i++) {
		int verIndex = verIds[i];

		if (ver.invM[verIndex] == 0.0f) {
			continue;
		}
		float3 delta = mul(gradients[i], -s * ver.invM[verIndex] * weight);
		if (!isfinite(delta.x) || !isfinite(delta.y) || !isfinite(delta.z)) {
			continue;
		}
		ver.p[verIndex] = add(ver.p[verIndex], delta);
	}

}
__global__ void initDampingVariablesKernel(DampingDevice damp, float* d_totalMass) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		*damp.poscm = make_float3(0, 0, 0);
		*damp.vcm = make_float3(0, 0, 0);
		*damp.omega = make_float3(0, 0, 0);
		*damp.angularMomentum = make_float3(0.0f, 0.0f, 0.0f);
		*d_totalMass = 0.0f;
		for (int i = 0; i < 9; i++) {
			damp.inertia[i] = 0.0f;
		}
	}
}

__global__ void computeDampingKernel(VertexDevice ver, DampingDevice damp, float* d_totalMass) {
	__shared__ float warpScalarSums[32];
	__shared__ float3 warpVectorSums[32];

	int vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;

	float3 weightedPosition = make_float3(0.0f, 0.0f, 0.0f);
	float3 weightedVelocity = make_float3(0.0f, 0.0f, 0.0f);
	float totalMass = 0.0f;

	if (vertexIndex < ver.N && ver.invM[vertexIndex] > 0.0f) {
		float mass = 1.0f / ver.invM[vertexIndex];

		weightedPosition = mul(ver.pos[vertexIndex], mass);
		weightedVelocity = mul(ver.v[vertexIndex], mass);
		totalMass = mass;
	}

	float3 blockWeightedPosition = blockReduceSum(weightedPosition, warpVectorSums);
	float3 blockWeightedVelocity = blockReduceSum(weightedVelocity, warpVectorSums);
	float blockTotalMass = blockReduceSum(totalMass, warpScalarSums);

	if (threadIdx.x == 0) {
		damp.centerPartials[blockIdx.x].weightedPosition = blockWeightedPosition;
		damp.centerPartials[blockIdx.x].weightedVelocity = blockWeightedVelocity;
		damp.centerPartials[blockIdx.x].totalMass = blockTotalMass;
	}
}

__global__ void finalizeCenterOfMassKernel(DampingDevice damp, float* d_totalMass) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		float3 weightedPosition = make_float3(0.0f, 0.0f, 0.0f);
		float3 weightedVelocity = make_float3(0.0f, 0.0f, 0.0f);

		float totalMass = 0.0f;

		for (int partialIndex = 0; partialIndex < damp.partialCount; partialIndex++) {
			weightedPosition = add(
				weightedPosition,
				damp.centerPartials[partialIndex].weightedPosition
			);

			weightedVelocity = add(
				weightedVelocity,
				damp.centerPartials[partialIndex].weightedVelocity
			);

			totalMass += damp.centerPartials[partialIndex].totalMass;
		}

		*d_totalMass = totalMass;

		if (totalMass <= 1e-8f || !isfinite(totalMass)) {
			*damp.poscm = make_float3(0.0f, 0.0f, 0.0f);
			*damp.vcm = make_float3(0.0f, 0.0f, 0.0f);
			*damp.omega = make_float3(0.0f, 0.0f, 0.0f);
			*damp.angularMomentum = make_float3(0.0f, 0.0f, 0.0f);

			for (int i = 0; i < 9; i++) {
				damp.inertia[i] = 0.0f;
			}

			return;
		}

		float inverseTotalMass = 1.0f / totalMass;

		*damp.poscm = mul(weightedPosition, inverseTotalMass);
		*damp.vcm = mul(weightedVelocity, inverseTotalMass);
	}
}

__global__ void computeAngularDampingKernel(VertexDevice ver, DampingDevice damp) {
	__shared__ float3 warpVectorSums[32];
	__shared__ Float3x3 warpTensorSums[32];

	int vertexIndex = blockIdx.x * blockDim.x + threadIdx.x;

	float3 angularMomentum = make_float3(0.0f, 0.0f, 0.0f);
	Float3x3 inertia = makeZeroFloat3x3();

	if (vertexIndex < ver.N && ver.invM[vertexIndex] > 0.0f) {
		float mass = 1.0f / ver.invM[vertexIndex];

		float3 poscm = *damp.poscm;
		float3 vcm = *damp.vcm;

		float3 r = sub(ver.pos[vertexIndex], poscm);
		float3 relativeVelocity = sub(ver.v[vertexIndex], vcm);

		angularMomentum = cross(r, mul(relativeVelocity, mass));

		float squaredRadius = dot(r, r);

		inertia.row0 = make_float3(mass * (squaredRadius - r.x * r.x), mass * (0.0f - r.x * r.y), mass * (0.0f - r.x * r.z));
		inertia.row1 = make_float3(mass * (0.0f - r.y * r.x), mass * (squaredRadius - r.y * r.y), mass * (0.0f - r.y * r.z));
		inertia.row2 = make_float3(mass * (0.0f - r.z * r.x), mass * (0.0f - r.z * r.y), mass * (squaredRadius - r.z * r.z));
	}

	float3 blockAngularMomentum = blockReduceSum(angularMomentum, warpVectorSums);
	Float3x3 blockInertia = blockReduceSum(inertia, warpTensorSums);
	if (threadIdx.x == 0) {
		damp.angularPartials[blockIdx.x].angularMomentum = blockAngularMomentum;

		damp.angularPartials[blockIdx.x].inertia[0] = blockInertia.row0.x;
		damp.angularPartials[blockIdx.x].inertia[1] = blockInertia.row0.y;
		damp.angularPartials[blockIdx.x].inertia[2] = blockInertia.row0.z;

		damp.angularPartials[blockIdx.x].inertia[3] = blockInertia.row1.x;
		damp.angularPartials[blockIdx.x].inertia[4] = blockInertia.row1.y;
		damp.angularPartials[blockIdx.x].inertia[5] = blockInertia.row1.z;

		damp.angularPartials[blockIdx.x].inertia[6] = blockInertia.row2.x;
		damp.angularPartials[blockIdx.x].inertia[7] = blockInertia.row2.y;
		damp.angularPartials[blockIdx.x].inertia[8] = blockInertia.row2.z;
	}
}

__global__ void finalizeOmegaKernel(DampingDevice damp) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		float3 angularMomentum = make_float3(0.0f, 0.0f, 0.0f);
		float inertiaValues[9];

		for (int i = 0; i < 9; i++) {
			inertiaValues[i] = 0.0f;
		}

		for (int partialIndex = 0; partialIndex < damp.partialCount; partialIndex++) {
			angularMomentum = add(angularMomentum, damp.angularPartials[partialIndex].angularMomentum);

			for (int inertiaIndex = 0; inertiaIndex < 9; inertiaIndex++) {
				inertiaValues[inertiaIndex] += damp.angularPartials[partialIndex].inertia[inertiaIndex];
			}
		}

		*damp.angularMomentum = angularMomentum;

		for (int i = 0; i < 9; i++) {
			damp.inertia[i] = inertiaValues[i];
		}

		mat3 inertia;
		inertia.row0 = make_float3(inertiaValues[0], inertiaValues[1], inertiaValues[2]);
		inertia.row1 = make_float3(inertiaValues[3], inertiaValues[4], inertiaValues[5]);
		inertia.row2 = make_float3(inertiaValues[6], inertiaValues[7], inertiaValues[8]);

		const float epsilon = 1e-6f;

		inertia.row0.x += epsilon;
		inertia.row1.y += epsilon;
		inertia.row2.z += epsilon;

		float determinant = det(inertia);

		if (fabsf(determinant) <= 1e-10f || !isfinite(determinant)) {
			*damp.omega = make_float3(0.0f, 0.0f, 0.0f);
			return;
		}

		if (!isfinite(angularMomentum.x) || !isfinite(angularMomentum.y) || !isfinite(angularMomentum.z)) {
			*damp.omega = make_float3(0.0f, 0.0f, 0.0f);
			return;
		}

		*damp.omega = mul(inverse(inertia), angularMomentum);

		if (!isfinite(damp.omega->x) || !isfinite(damp.omega->y) || !isfinite(damp.omega->z)) {
			*damp.omega = make_float3(0.0f, 0.0f, 0.0f);
		}
	}
}

__global__ void applyDampingKernel(VertexDevice ver, DampingDevice damp, float k_damping) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (verIndex >= ver.N) {
		return;
	}

	if (ver.invM[verIndex] == 0.0f) {
		return;
	}

	float k = fminf(fmaxf(k_damping, 0.0f), 1.0f);

	float3 poscm = *damp.poscm;
	float3 vcm = *damp.vcm;
	float3 omega = *damp.omega;

	if (
		!isfinite(poscm.x) || !isfinite(poscm.y) || !isfinite(poscm.z) ||
		!isfinite(vcm.x) || !isfinite(vcm.y) || !isfinite(vcm.z) ||
		!isfinite(omega.x) || !isfinite(omega.y) || !isfinite(omega.z)
		) {
		return;
	}

	float3 r = sub(ver.pos[verIndex], poscm);
	float3 rigidVelocity = add(vcm, cross(omega, r));

	float3 deltaV = sub(rigidVelocity, ver.v[verIndex]);
	ver.v[verIndex] = add(ver.v[verIndex], mul(deltaV, k));
}

__global__ void estimatePKernel(VertexDevice ver, float tstep) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIndex >= ver.N) return;
	ver.p[verIndex] = add(ver.pos[verIndex], mul(ver.v[verIndex], tstep));
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
namespace pbd {
	struct StretchProjector {
		__device__ __forceinline__ void operator()(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount) const {
			projectStretchConstraint(ver, cons, consIndex, tstep, iterationCount);
		}
	};
	struct BendingProjector {
		__device__ __forceinline__ void operator()(VertexDevice ver, ConstraintDevice cons, int consIndex, float tstep, int iterationCount) const {
			projectBendingConstraint(ver, cons, consIndex, tstep, iterationCount);
		}
	};

}

void solve(VertexDevice ver, ConstraintDevice cons, DampingDevice damp, CudaConstraintGraph& constraintIterationGraph, std::vector<float*>& vertexSet, std::vector<float*>& prevVertexSet, std::vector<unsigned int*>& indexSet, std::vector<int>& indexSetN, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, unsigned int* d_gridHashes, float* d_totalMass, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float cellSize, float selfThickness, float selfStiffness, int gridCapacity, float3* forces, float k_damping, float tstep, float currentTime, int iterationCount, int forceCount, int n, std::vector<int>& stretchColorOffset, std::vector<int>& bendingColorOffset, const float friction, const float restitution) {
	int threads = 256;
	int blocks = (ver.N + threads - 1) / threads;
	
	applyForceKernel<<<blocks, threads>>> (ver, forces, forceCount, tstep);
	checkCudaKernel("applyForceKernel launch failed");
	initDampingVariablesKernel << <1, 1 >> > (damp, d_totalMass);
	computeDampingKernel << <blocks, threads >> > (ver, damp, d_totalMass);
	finalizeCenterOfMassKernel << <1, 1 >> > (damp, d_totalMass);
	checkCudaKernel("finalizeCenterOfMassKernel launch failed");
	computeAngularDampingKernel << <blocks, threads >> > (ver, damp);
	checkCudaKernel("computeAngularDampingKernel launch failed");
	finalizeOmegaKernel << <1, 1 >> > (damp);
	checkCudaKernel("finalizeOmegaKernel launch failed");
	applyDampingKernel << <blocks, threads >> > (ver, damp, k_damping);
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

	if (!constraintIterationGraph.isInitialized()) {
		constraintIterationGraph.build([&](cudaStream_t stream) {
			for (int color = 0; color < cons.stretch.color.colorCount; color++) {
				int start = stretchColorOffset[color];
				int end = stretchColorOffset[color + 1];
				int count = end - start;

				if (count <= 0) continue;

				int colorBlocks = (count + threads - 1) / threads;

				solveStretchColorKernel << <colorBlocks, threads, 0, stream >> > (ver, cons, cons.stretch.color.constraintIds + start, count, tstep, iterationCount, pbd::StretchProjector{});
			}

			for (int color = 0; color < cons.bending.color.colorCount; color++) {
				int start = bendingColorOffset[color];
				int end = bendingColorOffset[color + 1];
				int count = end - start;

				if (count <= 0) continue;

				int colorBlocks = (count + threads - 1) / threads;

				solveBendingColorKernel << <colorBlocks, threads, 0, stream >> > (ver, cons, cons.bending.color.constraintIds + start, count, tstep, iterationCount, pbd::BendingProjector{});
			}
			checkCuda(cudaMemsetAsync(cons.collision.n, 0, sizeof(int), stream), "cudaMemsetAsync collision.n failed");
			});
	}

	for (int iter = 0; iter < iterationCount; iter++) {		
		constraintIterationGraph.launch();
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
			projectCollisionConstraint << <staticCollBlocks, threads >> > (ver, cons, 0);
			projectSelfCollisionConstraintKernel << <selfCollBlocks, threads >> > (ver, cons);
			
			applyAverageDeltaToPredictedKernel << <blocks, threads >> > (ver);
			checkCudaKernel("applyAverageDeltaToPredictedKernel launch failed");
		}
	}
	updateVerticesKernel<<<blocks, threads>>>(ver, tstep);
	checkCudaKernel("updateVerticesKernel launch failed");
	velocityUpdateKernel << <blocks, threads >> > (ver, cons, friction, restitution);
	checkCudaKernel("velocityUpdateKernel launch failed");
}