#include "CommonSolver.cuh"

__device__ void calcCentralDiff(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float tstep, Type type, float3& result) {
	float eps = 1e-4; // tstep은 너무커서 더 작은 값으로 중심차분법을 계산함
	if (eps <= 0.0f) { result = make_float3(0, 0, 0); return; }
	float3 advP = make_float3(0, 0, 0);
	float3 prevP = make_float3(0, 0, 0);
	float3 curP = make_float3(0, 0, 0);
	float advF = 0;
	float prevF = 0;
	
	for (int i = 0; i < 3; i++) {
		set(curP, i, get(ver.p[verIndex], i));
	}

	for (int j = 0; j < 3; j++) {
		// 편미분은 각 성분에 대해 독립적으로 적용되어야 하므로 
		// 기존에 변경한 내용을 초기화시켜야함
		for (int i = 0; i < 3; i++) {
			set(advP, i, get(curP, i));
			set(prevP, i, get(curP, i));
		}
		set(advP, j, get(curP, j) + eps);
		set(prevP, j, get(curP, j) - eps);

		switch (type) {
		case Stretch:
			advF = calcStretchOverride(ver, cons, consIndex, verIndex, advP);
			prevF = calcStretchOverride(ver, cons, consIndex, verIndex, prevP);
			break;
		case Bending:
			advF = calcBendingOverride(ver, cons, consIndex, verIndex, advP);
			prevF = calcBendingOverride(ver, cons, consIndex, verIndex, prevP);
		}
		float temp = (advF-prevF)/ (2 * eps);
		set(result, j, temp);
	}
}
__device__ void calcStretchGradient(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float3& result) {
	int2 vertices = cons.stretch.ver[consIndex];
	float3 p0 = ver.p[vertices.x];
	float3 p1 = ver.p[vertices.y];
	float3 d = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
	float d_len = length(d);
	if (d_len < 1e-8f || !isfinite(d_len)) {
		result = make_float3(0, 0, 0);
		return;
	}
	if (verIndex == vertices.x) {
		result = make_float3(-d.x / d_len, -d.y / d_len, -d.z / d_len);
	}
	else if(verIndex == vertices.y) {
		result = make_float3(d.x / d_len, d.y / d_len, d.z / d_len);
	}
	else {
		result = make_float3(0, 0, 0);
	}
}
__device__ void calcGradient(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float tstep, Type type, float3& result) {
	switch (type) {
	case Stretch:
	{
		calcStretchGradient(ver, cons, verIndex, consIndex, result);
		break;
	}
	case Bending: {
		//calcBendingGradient();
		calcCentralDiff(ver, cons, verIndex, consIndex, tstep, type, result);
		break;
	}
	default: {
		calcCentralDiff(ver, cons, verIndex, consIndex, tstep, type, result);
		break;
	}
	}
	
}

__device__ float stretch_impl(VertexDevice ver, ConstraintDevice cons, int consIndex, int overrideVid, const float3& overridePos) {
	int2 ids = cons.stretch.ver[consIndex];
	float3 p0 = pickP(ver, ids.x, overrideVid, overridePos);
	float3 p1 = pickP(ver, ids.y, overrideVid, overridePos);

	float len = length(sub(p1, p0));
	float l0 = cons.stretch.l0[consIndex];
	return len - l0;
}
__device__ float calcStretch(VertexDevice ver, ConstraintDevice cons, int consIndex) {
	return stretch_impl(ver, cons, consIndex, -1, make_float3(0, 0, 0));

}
__device__ float calcStretchOverride(VertexDevice ver, ConstraintDevice cons, int consIndex, int verIndex, const float3& newPos) {
	return stretch_impl(ver, cons, consIndex, verIndex, newPos);
}
__device__ float bending_impl(VertexDevice ver, ConstraintDevice cons, int consIndex, int overrideVid, const float3& overridePos) {
	int4 ids = cons.bending.ver[consIndex];
	float3 p0 = pickP(ver, ids.x, overrideVid, overridePos);
	float3 p1 = pickP(ver, ids.y, overrideVid, overridePos);
	float3 p2 = pickP(ver, ids.z, overrideVid, overridePos);
	float3 p3 = pickP(ver, ids.w, overrideVid, overridePos);
	float3 p1_0 = make_float3(0, 0, 0);
	float3 p2_0 = make_float3(0, 0, 0);
	float3 p3_0 = make_float3(0, 0, 0);
	float3 firstTerm = make_float3(0, 0, 0);
	float3 secondTerm = make_float3(0, 0, 0);

	float phi0 = cons.bending.phi0[consIndex];

	p1_0 = sub(p1, p0);
	p2_0 = sub(p2, p0);
	p3_0 = sub(p3, p0);

	firstTerm = cross(p1_0, p2_0);
	float firstLength = length(firstTerm);
	if (firstLength < 1e-8f) {
		return 0;
	}
	firstTerm = normalize(firstTerm);

	secondTerm = cross(p1_0, p3_0);
	float secondLength = length(secondTerm);
	if (secondLength < 1e-8f) {
		return 0;
	}
	secondTerm = normalize(secondTerm);


	float dotTerm = dot(firstTerm, secondTerm);
	dotTerm = clamp(dotTerm, -1.0f, 1.0f); //부동 소수점 오차로 인한 오류 방지

	//printf("angle=%f\n", acosf(dotTerm));
	return acosf(dotTerm) - phi0;
}
__device__ float calcBending(VertexDevice ver, ConstraintDevice cons, int consIndex) {
	return bending_impl(ver, cons, consIndex, -1, make_float3(0, 0, 0));

}
__device__ float calcBendingOverride(VertexDevice ver, ConstraintDevice cons, int consIndex, int verIndex, const float3& newPos) {
	return bending_impl(ver, cons, consIndex, verIndex, newPos);
}
__device__ float3 calcPoscm(VertexDevice ver) {
	float3 poscm = make_float3(0, 0, 0);
	float3 sumPosm = make_float3(0, 0, 0);
	float sumM = 0;
	int size = ver.N;
	for (int i = 0; i < size; i++) {
		if (ver.invM[i] == 0) continue;
		float m = 1 / ver.invM[i];
		for (int j = 0; j < 3; j++) {
			float item = get(sumPosm, j) + get(ver.pos[i], j) * m;
			set(sumPosm, j, item);
		}
		sumM += m;
	}
	for (int i = 0; i < 3; i++) {
		set(poscm, i, get(sumPosm, i) / sumM);
	}
	return poscm;
}
__device__ float3 calcVcm(VertexDevice ver) {
	float3 vcm = make_float3(0, 0, 0);
	float3 sumVm = make_float3(0, 0, 0);
	float sumM = 0;
	int size = ver.N;
	for (int i = 0; i < size; i++) {
		if (ver.invM[i] == 0) continue;
		float m = 1 / ver.invM[i];
		for (int j = 0; j < 3; j++) {
			float item = get(sumVm, j) + get(ver.v[i], j) * m;
			set(sumVm, j, item);
		}
		sumM += m;
	}
	for (int i = 0; i < 3; i++) {
		set(vcm, i, get(sumVm, i) / sumM);
	}
	return vcm;
}
__device__ float3 calcOmega(VertexDevice ver, float3 pcm) {
	float3 poscm = make_float3(pcm.x, pcm.y, pcm.z);
	int size = ver.N;
	float3 L = make_float3(0, 0, 0);
	float3 omega = make_float3(0, 0, 0);
	mat3 I;
	I.row0 = make_float3(1, 0, 0);
	I.row1 = make_float3(0, 1, 0);
	I.row2 = make_float3(0, 0, 1);
	for (int i = 0; i < size; i++) {
		float3 r = make_float3(0, 0, 0);
		float3 velocity = make_float3(0, 0, 0);
		float3 verPos = ver.pos[i];
		if (ver.invM[i] == 0) continue;
		float m = 1 / ver.invM[i];
		for (int j = 0; j < 3; j++) {
			set(r, j, get(verPos, j) - get(poscm, j));
			float item = m * get(ver.v[i], j);
			set(velocity, j, item);
		}
		float3 tmp = make_float3(0, 0, 0);
		tmp = cross(r, velocity);
		L = add(L, tmp);
		mat3 skew;
		skew.row0 = make_float3(0, get(r, 2), -get(r, 1));
		skew.row1 = make_float3(-get(r, 2), 0, get(r, 0));
		skew.row2 = make_float3(get(r, 1), -get(r, 0), 0);
		I = add(I, mul(mul(skew, transpose(skew)), m));
	};
	omega = mul(inverse(I), L);

	return omega;
}

__global__ void clearForceKernel(float3* extForce, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) {
		return;
	}
	extForce[i] = make_float3(0, 0, 0);
}
__global__ void windForceKernel(VertexDevice ver, int3* tris, int triCount, float3* extForce, float3 windVelocity, float airCoeff) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= triCount) {
		return;
	}
	int a = tris[i].x;
	int b = tris[i].y;
	int c = tris[i].z;

	float3 posa = ver.pos[a];
	float3 posb = ver.pos[b];
	float3 posc = ver.pos[c];

	float3 va = ver.v[a];
	float3 vb = ver.v[b];
	float3 vc = ver.v[c];

	float3 e1 = sub(posb, posa);
	float3 e2 = sub(posc, posa);

	float3 n = cross(e1, e2);

	float area2 = length(n);
	if (area2 < 1e-8f) {
		return;
	}
	float area = 0.5f * area2;

	float3 normal = mul(n, 1 / area2);

	float3 tmp = add(va, vb);
	tmp = add(tmp, vc);
	float3 vTri = mul(tmp, 1.0f / 3.0f);
	float3 vRel = sub(windVelocity, vTri);

	float vn = dot(vRel, normal);

	if (vn <= 0.0f) return;

	float3 Ftri = mul(normal, airCoeff * area * vn);

	float3 each = mul(Ftri, 1.0f / 3.0f);

}

__device__ void createCollisionConstraint(ConstraintDevice cons, int3 tri, int verIndex, float k, float thickness, float3 q, float3 normal) {
	int idx = atomicAdd(cons.collision.n, 1);

	if (idx >= cons.collision.capacity) {
		return;
	}
	cons.collision.tri[idx] = tri;
	cons.collision.ver[idx] = verIndex;
	cons.collision.k[idx] = k;
	cons.collision.thickness[idx] = thickness;
	cons.collision.q[idx] = q;
	cons.collision.normal[idx] = normal;

}
__global__ void detectCollisionKernel(VertexDevice ver, ConstraintDevice cons, float* triangle, unsigned int* d_triangleIndices, int triangleN) {
	int verIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIdx >= ver.N) {
		return;
	}
	if (ver.invM[verIdx] == 0.0f) {
		return;
	}

	float3 x = ver.pos[verIdx];
	float3 p = ver.p[verIdx];
	float3 ray = sub(p, x);

	const float thickness = 0.0025f;
	const float eps = 1e-6f;

	float bestMeasure = 1e30f;
	int3 bestTri = make_int3(0, 0, 0);
	float3 bestQ = make_float3(0.0f, 0.0f, 0.0f);
	float3 bestN = make_float3(0.0f, 0.0f, 0.0f);
	bool found = false;

	for (int i = 0; i < triangleN; i++) {
		unsigned int i0 = d_triangleIndices[3 * i + 0];
		unsigned int i1 = d_triangleIndices[3 * i + 1];
		unsigned int i2 = d_triangleIndices[3 * i + 2];

		float3 p0 = make_float3(triangle[3 * i0 + 0], triangle[3 * i0 + 1], triangle[3 * i0 + 2]);
		float3 p1 = make_float3(triangle[3 * i1 + 0], triangle[3 * i1 + 1], triangle[3 * i1 + 2]);
		float3 p2 = make_float3(triangle[3 * i2 + 0], triangle[3 * i2 + 1], triangle[3 * i2 + 2]);

		float3 triNormal = cross(sub(p1, p0), sub(p2, p0));
		float nLen = length(triNormal);
		if (nLen <= 1e-8f) {
			continue;
		}
		triNormal = mul(triNormal, 1.0f / nLen);

		float denom = dot(ray, triNormal);
		float distP = dot(sub(p, p0), triNormal);

		bool hitBySegment = false;
		float3 contactPoint = make_float3(0.0f, 0.0f, 0.0f);
		float measure = 1e30f;

		if (fabsf(denom) > 1e-8f) {
			float t = dot(sub(p0, x), triNormal) / denom;
			if (t > 0.0f && t < 1.0f) {
				float3 intersectPoint = add(x, mul(ray, t));

				float3 barycentricPoint = barycentric(p0, p1, p2, intersectPoint);
				float u = barycentricPoint.x;
				float v = barycentricPoint.y;
				float w = barycentricPoint.z;

				bool inside =
					u >= -1e-6f && v >= -1e-6f && w >= -1e-6f &&
					u <= 1.0f + 1e-6f && v <= 1.0f + 1e-6f && w <= 1.0f + 1e-6f &&
					fabsf((u + v + w) - 1.0f) <= 1e-4f;

				if (inside) {
					hitBySegment = true;
					contactPoint = intersectPoint;
					measure = t;
				}
			}
		}

		bool hitByPenetration = false;
		if (!hitBySegment) {
			float3 projP = sub(p, mul(triNormal, distP));

			float3 barycentricPoint = barycentric(p0, p1, p2, projP);
			float u = barycentricPoint.x;
			float v = barycentricPoint.y;
			float w = barycentricPoint.z;

			bool inside =
				u >= -1e-6f && v >= -1e-6f && w >= -1e-6f &&
				u <= 1.0f + 1e-6f && v <= 1.0f + 1e-6f && w <= 1.0f + 1e-6f &&
				fabsf((u + v + w) - 1.0f) <= 1.0e-4f;

			if (inside) {
				if (distP < thickness) {
					hitByPenetration = true;
					contactPoint = projP;
					measure = distP;
				}
			}
		}

		if (hitBySegment || hitByPenetration) {
			if (measure < bestMeasure) {
				bestMeasure = measure;
				bestTri = make_int3(i0, i1, i2);
				bestQ = contactPoint;
				bestN = triNormal;
				found = true;
			}
		}
	}

	if (found) {
		createCollisionConstraint(cons, bestTri, verIdx, 1.0f, thickness, bestQ, bestN);
	}
}

__device__ bool triContainsVertex(const int3& tri, int v) {
	return tri.x == v || tri.y == v || tri.z == v;
}
__device__ float3 closestPointOnSegment(const float3& p, const float3& a, const float3& b) {
	float3 ab = sub(a, b);
	float denom = dot(ab, ab);
	if (denom <= 1e-12f) {
		return a;
	}
	float t = dot(sub(p, a), ab) / denom;
	t = clamp(t, 0.0f, 1.0f);
	return add(a, mul(ab, t));
}
__device__ float3 closestPointOnTriangle(const float3& p, const float3& a, const float3& b, const float3& c) {
	float3 ab = sub(b, a);
	float3 ac = sub(c, a);
	float3 ap = sub(p, a);

	float d1 = dot(ab, ap);
	float d2 = dot(ac, ap);

	if (d1 <= 0.0f && d2 <= 0.0f) {
		return a;
	}

	float3 bp = sub(p, b);
	float d3 = dot(ab, bp);
	float d4 = dot(ac, bp);

	if (d3 >= 0.0f && d4 <= d3) {
		return b;
	}

	float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		return add(a, mul(ab, v));
	}

	float3 cp = sub(p, c);
	float d5 = dot(ab, cp);
	float d6 = dot(ac, cp);

	if (d6 >= 0.0f && d5 <= d6) {
		return c;
	}

	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float w = d2 / (d2 - d6);
		return add(a, mul(ac, w));
	}
	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		float3 bc = sub(c, b);
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return add(b, mul(bc, w));
	}
	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;

	return add(a, add(mul(ab, v), mul(ac, w)));
}
__device__ void findAndCreateSelfCollisionForVertex(VertexDevice ver, ConstraintDevice cons, int queryVertex, int anchorVertex, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float thickness, float stiffness) {
	if (ver.invM[queryVertex] == 0.0f) {
		return;
	}
	float3 queryPos = ver.p[queryVertex];
	float bestDist2 = thickness * thickness;
	bool found = false;

	int3 bestTri = make_int3(0, 0, 0);
	float3 bestQ = make_float3(0.0f, 0.0f, 0.0f);
	float3 bestN = make_float3(0.0f, 0.0f, 0.0f);

	int begin = vertTriOffset[anchorVertex];
	int end = vertTriOffset[anchorVertex + 1];

	for (int i = begin; i < end; i++) {
		int triIndex = vertTriArray[i];
		int4 tri4 = selfTris[triIndex];
		int3 tri = make_int3(tri4.x, tri4.y, tri4.z);

		if (triContainsVertex(tri, queryVertex)) {
			continue;
		}

		float3 p0 = ver.p[tri.x];
		float3 p1 = ver.p[tri.y];
		float3 p2 = ver.p[tri.z];

		float3 triNormal = cross(sub(p1, p0), sub(p2, p0));
		float nLen = length(triNormal);
		if (nLen <= 1e-8f) {
			continue;
		}
		triNormal = normalize(triNormal);

		float3 q = closestPointOnTriangle(queryPos, p0, p1, p2);
		float3 diff = sub(queryPos, q);
		float dist2 = dot(diff, diff);

		if (dist2 >= bestDist2) {
			continue;
		}
		float3 n;
		if (dist2 > 1e-12f) {
			float invLen = rsqrtf(dist2);
			n = mul(diff, invLen);
		}
		else {
			float side = dot(sub(queryPos, p0), triNormal);
			n = (side >= 0.0f) ? triNormal : mul(triNormal, -1.0f);
		}
		bestDist2 = dist2;
		bestTri = tri;
		bestQ = q;
		bestN = n;
		found = true;
	}
	if (found) {
		createCollisionConstraint(cons, bestTri, queryVertex, stiffness, thickness, bestQ, bestN);
	}
}

__global__ void buildSelfPairsGPUKernel(VertexDevice ver, int2* outPairs, int* pairCount, int maxPairs, float cellSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= ver.N || j >= ver.N || i >= j) return;

	float3 diff = sub(ver.p[i], ver.p[j]);
	float dist2 = dot(diff, diff);
	if (dist2 < cellSize * cellSize) {
		// [MODIFIED] atomicAdd를 이용해 GPU 전역에서 안전하게 인덱스를 세고 데이터를 채움
		int idx = atomicAdd(pairCount, 1);
		if (idx < maxPairs) {
			outPairs[idx] = make_int2(i, j);
		}
	}
}
__global__ void detectSelfCollisionKernel(VertexDevice ver, ConstraintDevice cons, const int2* selfPairs, int* d_pairCount, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float thickness, float stiffness) {
	int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;

	// [MODIFIED] GPU 전역 메모리에 저장된 실제 쌍의 개수를 읽어와 비교
	if (pairIdx >= *d_pairCount) {
		return;
	}

	int2 pair = selfPairs[pairIdx];
	int a = pair.x;
	int b = pair.y;
	if (a < 0 || b < 0 || a >= ver.N || b >= ver.N) {
		return;
	}
	findAndCreateSelfCollisionForVertex(ver, cons, a, b, selfTris, vertTriArray, vertTriOffset, thickness, stiffness);
	findAndCreateSelfCollisionForVertex(ver, cons, b, a, selfTris, vertTriArray, vertTriOffset, thickness, stiffness);
}
void launchBuildSelfPairs(VertexDevice ver, int2* outPairs, int* pairCount, int maxPairs, float cellSize) {
	dim3 pairBlocks((ver.N + 15) / 16, (ver.N + 15) / 16);
	dim3 pairThreads(16, 16);

	buildSelfPairsGPUKernel << <pairBlocks, pairThreads >> > (ver, outPairs, pairCount, maxPairs, cellSize);
}
__global__ void updateFloorKernel(float* d_floorVertices, float time) {
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		float floorBaseY = -1.5f;
		float animatedFloorY = floorBaseY + 0.5f * sin(time * 1.0f);

		// 4개 정점의 Y좌표(index 1, 4, 7, 10)를 직접 수정
		d_floorVertices[1] = animatedFloorY;
		d_floorVertices[4] = animatedFloorY;
		d_floorVertices[7] = animatedFloorY;
		d_floorVertices[10] = animatedFloorY;
	}
}
void launchUpdateFloor(float* d_floorVertices, float time) {
	updateFloorKernel << <1, 1 >> > (d_floorVertices, time);
}