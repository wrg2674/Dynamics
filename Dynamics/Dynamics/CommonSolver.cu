#include "CommonSolver.cuh"


__device__ void calcCentralDiff(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float tstep, Type type, float3& result) {
	float eps = 1e-5; // tstep은 너무커서 더 작은 값으로 중심차분법을 계산함
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
__device__ void calcBendingGradient(VertexDevice ver, ConstraintDevice cons, int consIndex, float3* result) {
	result[0] = make_float3(0, 0, 0);
	result[1] = make_float3(0, 0, 0);
	result[2] = make_float3(0, 0, 0);
	result[3] = make_float3(0, 0, 0);

	int4 ids = cons.bending.ver[consIndex];
	float3 p0 = ver.p[ids.x];
	float3 p1 = ver.p[ids.y];
	float3 p2 = ver.p[ids.z];
	float3 p3 = ver.p[ids.w];

	float3 e = sub(p1, p0);
	float eLen = length(e);
	if (eLen < 1e-6f) { 
		return;
	}
	float3 n1 = cross(e, sub(p2, p0));
	float3 n2 = cross(e, sub(p3, p0));
	float n1l2 = norm2(n1);
	float n2l2 = norm2(n2);

	if (n1l2 < 1e-8f || n2l2 < 1e-8f) {
		return;
	}

	float3 q2 = mul(n1, eLen / n1l2);
	float3 q3 = mul(n2, -eLen / n2l2);
	float invEL2 = 1.0f / (eLen * eLen);
	float3 q0 = add(mul(q2, dot(sub(p2, p1), e) * invEL2), mul(q3, dot(sub(p3, p1), e) * invEL2));
	float3 q1 = add(mul(q2, dot(sub(p3, p2), e) * invEL2), mul(q3, dot(sub(p0, p3), e) * invEL2));	result[0] = q0;
	result[1] = q1;
	result[2] = q2;
	result[3] = q3;
}
__device__ void calcGradient(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float tstep, Type type, float3& result) {
	switch (type) {
	case Stretch:
	{
		calcStretchGradient(ver, cons, verIndex, consIndex, result);
		break;
	}
	case Bending: {
		float3 grads[4];
		calcBendingGradient(ver, cons, consIndex, grads);
		int4 ids = cons.bending.ver[consIndex];
		if (verIndex == ids.x) {
			result = grads[0]; 
		}
		else if (verIndex == ids.y) {
			result = grads[1]; 
		}
		else if (verIndex == ids.z) {
			result = grads[2]; 
		}
		else if (verIndex == ids.w) {
			result = grads[3]; 
		}
		else {
			result = make_float3(0, 0, 0);
		}
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


	float3 e01 = sub(p1, p0);
	float edgeLen = length(e01);

	if (edgeLen < 1e-8f || !isfinite(edgeLen)) {
		return 0.0f;
	}
	float3 n1 = cross(sub(p2, p0), e01);
	float3 n2 = cross(e01, sub(p3, p0));

	float n1Len = length(n1);
	float n2Len = length(n2);

	if (n1Len < 1e-8f || n2Len < 1e-8f) {
		return 0.0f;
	}
	n1 = normalize(n1);
	n2 = normalize(n2);

	float dotTerm = dot(n1, n2);
	dotTerm = clamp(dotTerm, -0.9999f, 0.9999f);

	float phi = acosf(dotTerm);
	if (dot(cross(n1, n2), e01) > 0.0f) {
		phi = -phi;
	}
	float phi0 = cons.bending.phi0[consIndex];

	return phi - phi0;
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

__device__ void createCollisionConstraint(ConstraintDevice cons, int3 tri, int verIndex, float k, float thickness, float3 q, float3 normal, float3 colliderVelocity) {
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
	cons.collision.colliderVelocity[idx] = colliderVelocity;

}
__device__ void createSelfCollisionConstraint(ConstraintDevice cons, int3 tri, int verIndex, float k, float thickness, float3 q, float3 normal) {
	int idx = atomicAdd(cons.selfCollision.n, 1);
	if (idx >= cons.selfCollision.capacity) {
		return;
	}
	cons.selfCollision.tri[idx] = tri;
	cons.selfCollision.ver[idx] = verIndex;
	cons.selfCollision.k[idx] = k;
	cons.selfCollision.thickness[idx] = thickness;
	cons.selfCollision.q[idx] = q;
	cons.selfCollision.normal[idx] = normal;
}
__global__ void detectContinuousCollisionKernel(VertexDevice ver, ConstraintDevice cons, float* triangle, float* prevTriangle, unsigned int* d_triangleIndices, int triangleN, float tstep) {
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

	float bestT = 1e30f;
	int3 bestTri = make_int3(0, 0, 0);
	float3 bestQ = make_float3(0.0f, 0.0f, 0.0f);
	float3 bestN = make_float3(0.0f, 0.0f, 0.0f);
	float3 bestColliderVelocity = make_float3(0.0f, 0.0f, 0.0f);

	bool found = false;

	for (int i = 0; i < triangleN; i++) {
		unsigned int i0 = d_triangleIndices[3 * i + 0];
		unsigned int i1 = d_triangleIndices[3 * i + 1];
		unsigned int i2 = d_triangleIndices[3 * i + 2];

		float3 p0 = make_float3(triangle[3 * i0 + 0], triangle[3 * i0 + 1], triangle[3 * i0 + 2]);
		float3 p1 = make_float3(triangle[3 * i1 + 0], triangle[3 * i1 + 1], triangle[3 * i1 + 2]);
		float3 p2 = make_float3(triangle[3 * i2 + 0], triangle[3 * i2 + 1], triangle[3 * i2 + 2]);

		float3 prevP0 = make_float3(prevTriangle[3 * i0 + 0], prevTriangle[3 * i0 + 1], prevTriangle[3 * i0 + 2]);
		float3 prevP1 = make_float3(prevTriangle[3 * i1 + 0], prevTriangle[3 * i1 + 1], prevTriangle[3 * i1 + 2]);
		float3 prevP2 = make_float3(prevTriangle[3 * i2 + 0], prevTriangle[3 * i2 + 1], prevTriangle[3 * i2 + 2]);

		float3 triNormal = cross(sub(p1, p0), sub(p2, p0));
		float nLen = length(triNormal);
		if (nLen <= 1e-8f) {
			continue;
		}
		triNormal = mul(triNormal, 1.0f / nLen);

		float3 d0 = sub(p0, prevP0);
		float3 d1 = sub(p1, prevP1);
		float3 d2 = sub(p2, prevP2);

		float3 colliderDisplacement = mul(add(add(d0, d1), d2), 1.0f / 3.0f);

		float safeDt = fmaxf(tstep, 1e-8f);
		float3 colliderVelocity = mul(colliderDisplacement, 1.0f / safeDt);
		float3 xInCurrentColliderFrame = add(x, colliderDisplacement);
		float3 ray = sub(p, xInCurrentColliderFrame);

		float rayLen2 = dot(ray, ray);
		if (rayLen2 <= eps * eps) {
			continue;
		}
		float denom = dot(ray, triNormal);
		if (fabsf(denom) <= 1e-8f) {
			continue;
		}
		float t = dot(sub(p0, xInCurrentColliderFrame), triNormal) / denom;
		if (t < 0.0f || t>1.0f) {
			continue;
		}

		float3 intersectPoint = add(xInCurrentColliderFrame, mul(ray, t));

		float3 barycentricPoint = barycentric(p0, p1, p2, intersectPoint);
		float u = barycentricPoint.x;
		float v = barycentricPoint.y;
		float w = barycentricPoint.z;

		bool inside =
			u >= -1e-6f && v >= -1e-6f && w >= -1e-6f &&
			u <= 1.0f + 1e-6f && v <= 1.0f + 1e-6f && w <= 1.0f + 1e-6f &&
			fabsf((u + v + w) - 1.0f) <= 1e-4f;

		if (!inside) {
			continue;
		}
		float predictedDist = dot(sub(p, intersectPoint), triNormal);

		if (predictedDist >= thickness) {
			continue;
		}
		if (t < bestT) {
			bestT = t;
			bestTri = make_int3(i0, i1, i2);
			bestQ = intersectPoint;
			bestN = triNormal;
			bestColliderVelocity = colliderVelocity;
			found = true;
		}
	}

	if (found) {
		createCollisionConstraint(cons, bestTri, verIdx, 1.0f, thickness, bestQ, bestN, bestColliderVelocity);
	}
}
__global__ void detectStaticCollisionKernel(VertexDevice ver, ConstraintDevice cons, float* triangle, float* prevTriangle, unsigned int* d_triangleIndices, int triangleN, float tstep) {
	int verIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (verIdx >= ver.N) {
		return;
	}

	if (ver.invM[verIdx] == 0.0f) {
		return;
	}
	float3 p = ver.p[verIdx];

	const float thickness = 0.0025f;

	float bestDist = 1e30f;
	int3 bestTri = make_int3(0, 0, 0);
	float3 bestQ = make_float3(0.0f, 0.0f, 0.0f);
	float3 bestN = make_float3(0.0f, 0.0f, 0.0f);
	float3 bestColliderVelocity = make_float3(0.0f, 0.0f, 0.0f);
	bool found = false;
	for (int i = 0; i < triangleN; i++) {
		unsigned int i0 = d_triangleIndices[3 * i + 0];
		unsigned int i1 = d_triangleIndices[3 * i + 1];
		unsigned int i2 = d_triangleIndices[3 * i + 2];

		float3 p0 = make_float3(triangle[3 * i0 + 0], triangle[3 * i0 + 1], triangle[3 * i0 + 2]);
		float3 p1 = make_float3(triangle[3 * i1 + 0], triangle[3 * i1 + 1], triangle[3 * i1 + 2]);
		float3 p2 = make_float3(triangle[3 * i2 + 0],triangle[3 * i2 + 1],triangle[3 * i2 + 2]);

		float3 prevP0 = make_float3(prevTriangle[3 * i0 + 0], prevTriangle[3 * i0 + 1], prevTriangle[3 * i0 + 2]);
		float3 prevP1 = make_float3(prevTriangle[3 * i1 + 0], prevTriangle[3 * i1 + 1], prevTriangle[3 * i1 + 2]);
		float3 prevP2 = make_float3(prevTriangle[3 * i2 + 0], prevTriangle[3 * i2 + 1], prevTriangle[3 * i2 + 2]);

		float3 triNormal = cross(sub(p1, p0), sub(p2, p0));
		float nLen = length(triNormal);

		if (nLen <= 1e-8f || !isfinite(nLen)) {
			continue;
		}

		triNormal = mul(triNormal, 1.0f / nLen);

		float signedDist = dot(sub(p, p0), triNormal);
		if (signedDist >= thickness) {
			continue;
		}
		float3 q = sub(p, mul(triNormal, signedDist));

		float3 barycentricPoint = barycentric(p0, p1, p2, q);
		float u = barycentricPoint.x;
		float v = barycentricPoint.y;
		float w = barycentricPoint.z;

		bool inside =
			u >= -1e-6f && v >= -1e-6f && w >= -1e-6f &&
			u <= 1.0f + 1e-6f && v <= 1.0f + 1e-6f && w <= 1.0f + 1e-6f &&
			fabsf((u + v + w) - 1.0f) <= 1e-4f;

		if (!inside) {
			continue;
		}

		if (signedDist < bestDist) {
			bestDist = signedDist;
			bestTri = make_int3(i0, i1, i2);
			bestQ = q;
			bestN = triNormal;

			float3 prevQ = add(add(mul(prevP0, u), mul(prevP1, v)), mul(prevP2, w));
			float safeDt = fmaxf(tstep, 1e-8f);
			bestColliderVelocity = mul(sub(q, prevQ), 1.0f / safeDt);
			found = true;
		}
	}
	if (found) {
		createCollisionConstraint(cons, bestTri, verIdx, 0.5f, thickness, bestQ, bestN, bestColliderVelocity);
	}
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
__global__ void detectSelfCollisionKernel(VertexDevice ver, ConstraintDevice cons, int* d_gridIndices, int* d_cellStart, int* d_cellEnd, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset, float cellSize, float thickness, float stiffness) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= ver.N) return;

	if (ver.invM[i] == 0.0f) {
		return;
	}

	float3 pos = ver.p[i];
	int3 cell = getCellCoords(pos, cellSize);

	float bestDist2 = thickness * thickness;
	bool found = false;

	int3 bestTri = make_int3(0, 0, 0);
	float3 bestQ = make_float3(0.0f, 0.0f, 0.0f);
	float3 bestN = make_float3(0.0f, 0.0f, 0.0f);

	for (int z = -1; z <= 1; z++) {
		for (int y = -1; y <= 1; y++) {
			for (int x = -1; x <= 1; x++) {
				int3 neighborCell = make_int3(cell.x + x, cell.y + y, cell.z + z);
				unsigned int hash = computeHash(neighborCell);

				int start = d_cellStart[hash];
				if (start == -1) {
					continue;
				}
				int end = d_cellEnd[hash];
				for (int j_idx = start; j_idx < end; j_idx++) {
					int j = d_gridIndices[j_idx];
					if (i == j) {
						continue;
					}
					if (i >= j) {
						continue;
					}
					int begin = vertTriOffset[j];
					int triEnd = vertTriOffset[j + 1];

					for (int t = begin; t < triEnd; t++) {
						int triIndex = vertTriArray[t];
						int4 tri4 = selfTris[triIndex];
						int3 tri = make_int3(tri4.x, tri4.y, tri4.z);
						//findAndCreateSelfCollisionForVertex(ver, cons, i, j, selfTris, vertTriArray, vertTriOffset, thickness, stiffness);
						if (triangleSharesOneRingOfVertex(tri, i, selfTris, vertTriArray, vertTriOffset)) {
							continue;
						}
						float3 p0 = ver.p[tri.x];
						float3 p1 = ver.p[tri.y];
						float3 p2 = ver.p[tri.z];

						float3 triNormal = cross(sub(p1, p0), sub(p2, p0));
						float nLen = length(triNormal);

						if (nLen <= 1e-8f || !isfinite(nLen)) {
							continue;
						}

						triNormal = mul(triNormal, 1.0f / nLen);

						float3 q = closestPointOnTriangle(pos, p0, p1, p2);
						float3 diff = sub(pos, q);
						float dist2 = dot(diff, diff);
						if (!isfinite(dist2)) {
							continue;
						}
						if (dist2 >= bestDist2) {
							continue;
						}
						float3 n;
						if (dist2 > 1e-12f) {
							float invLen = rsqrtf(dist2);
							n = mul(diff, invLen);
						}
						else {
							float side = dot(sub(pos, p0), triNormal);
							n = (side >= 0.0f) ? triNormal : mul(triNormal, -1.0f);
						}
						bestDist2 = dist2;
						bestTri = tri;
						bestQ = q;
						bestN = n;
						found = true;
					}
				}
			}
		}
	}
	if (found) {
		createSelfCollisionConstraint(cons, bestTri, i, stiffness, thickness, bestQ, bestN);
	}
}
__device__ bool triContainsVertex(const int3& tri, int v) {
	return tri.x == v || tri.y == v || tri.z == v;
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

		if (triangleSharesOneRingOfVertex(tri, queryVertex, selfTris, vertTriArray, vertTriOffset)) {
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
		createSelfCollisionConstraint(cons, bestTri, queryVertex, stiffness, thickness, bestQ, bestN);
	}
}

__global__ void buildSelfPairsGPUKernel(VertexDevice ver, int2* outPairs, int* pairCount, int maxPairs, float cellSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ver.N) {
		return;
	}
	float3 pos_i = ver.p[i];
	int3 cell = getCellCoords(pos_i, cellSize);
	float r2 = cellSize * cellSize;

	for (int j = i + 1; j < ver.N; j++) {
		float3 diff = sub(pos_i, ver.p[j]);
		float dist2 = dot(diff, diff);

		if (dist2 < r2) {
			int idx = atomicAdd(pairCount, 1);
			if (idx < maxPairs) {
				outPairs[idx] = make_int2(i, j);
			}
		}
	}
}

__device__ bool triangleSharesOneRingOfVertex(const int3& candidateTri, int queryVertex, const int4* selfTris, const int* vertTriArray, const int* vertTriOffset) {
	int begin = vertTriOffset[queryVertex];
	int end = vertTriOffset[queryVertex + 1];

	for (int i = begin; i < end; i++) {
		int incidentTriIndex = vertTriArray[i];
		int4 incidentTri4 = selfTris[incidentTriIndex];

		int a = incidentTri4.x;
		int b = incidentTri4.y;
		int c = incidentTri4.z;

		if (triContainsVertex(candidateTri, a)) {
			return true;
		}
		if (triContainsVertex(candidateTri, b)) {
			return true;
		}
		if (triContainsVertex(candidateTri, c)) {
			return true;
		}
	}
	return false;
}
void launchBuildSelfPairs(VertexDevice ver, int2* outPairs, int* pairCount, int maxPairs, float cellSize) {
	dim3 pairBlocks((ver.N + 15) / 16, (ver.N + 15) / 16);
	dim3 pairThreads(16, 16);

	buildSelfPairsGPUKernel << <pairBlocks, pairThreads >> > (ver, outPairs, pairCount, maxPairs, cellSize);
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
	float3 accel = add(mul(totalForce, invM), make_float3(0, -9.8, 0));
	curV = add(curV, mul(accel, tstep));
}
__global__ void applyAverageDeltaToPredictedKernel(VertexDevice ver) {
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
//32개의 스레드 값을 합치는데 뎃셈의 횟수를 줄이면서 수행함.
__device__ float warpReduceSum(float value) {
	for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
		// 스레드의 값을 다른 스레드에서 가져오는 함수 (공유 메모리 안거침)
		// 첫번째 매개변수 : 계산에 참여하는 warp의 스레드 번호(lane) 마스크
		// (0xffffffffu는 11111111 11111111 11111111 11111111으로 비트가 1이면 그 lane은 계산에 참여한다는 의미)
		// 두번째 매개변수 : lane끼리 교환할 변수명(단, 문자열이 아닌 변수로써 전달)
		// 세번째 매개변수 : 몇칸 뒤에 있는 lane의 값을 가져올지 지정
		value += __shfl_down_sync(0xffffffffu, value, offset);
	}

	return value;
}
__device__ float blockReduceSum(float value, float* warpSums) {
	int laneIndex = threadIdx.x & (warpSize - 1);
	int warpIndex = threadIdx.x / warpSize;
	int warpCount = (blockDim.x + warpSize - 1) / warpSize;

	value = warpReduceSum(value);

	if (laneIndex == 0) {
		warpSums[warpIndex] = value;
	}

	__syncthreads();

	float blockSum = 0.0f;

	if (warpIndex == 0) {
		if (laneIndex < warpCount) {
			blockSum = warpSums[laneIndex];
		}

		blockSum = warpReduceSum(blockSum);
	}
	__syncthreads();

	return blockSum;
}
__device__ float3 warpReduceSum(float3 value) {
	for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
		value.x += __shfl_down_sync(0xffffffffu, value.x, offset);
		value.y += __shfl_down_sync(0xffffffffu, value.y, offset);
		value.z += __shfl_down_sync(0xffffffffu, value.z, offset);
	}

	return value;
}
__device__ float3 blockReduceSum(float3 value, float3* warpSums) {
	int laneIndex = threadIdx.x & (warpSize - 1);
	int warpIndex = threadIdx.x / warpSize;
	int warpCount = (blockDim.x + warpSize - 1) / warpSize;

	value = warpReduceSum(value);

	if (laneIndex == 0) {
		warpSums[warpIndex] = value;
	}
	__syncthreads();
	float3 blockSum = make_float3(0.0f, 0.0f, 0.0f);

	if (warpIndex == 0) {
		if (laneIndex < warpCount) {
			blockSum = warpSums[laneIndex];
		}

		blockSum = warpReduceSum(blockSum);
	}
	__syncthreads();
	return blockSum;
}

struct Float3x3 {
	float3 row0;
	float3 row1;
	float3 row2;
};

__device__ Float3x3 makeFloat3x3(float3 row0, float3 row1, float3 row2) {
	Float3x3 value;
	value.row0 = row0;
	value.row1 = row1;
	value.row2 = row2;
	return value;
}

__device__ Float3x3 makeZeroFloat3x3() {
	return makeFloat3x3(
		make_float3(0.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 0.0f),
		make_float3(0.0f, 0.0f, 0.0f)
	);
}
__device__ Float3x3 warpReduceSum(Float3x3 value) {
	for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
		value.row0.x += __shfl_down_sync(0xffffffffu, value.row0.x, offset);
		value.row0.y += __shfl_down_sync(0xffffffffu, value.row0.y, offset);
		value.row0.z += __shfl_down_sync(0xffffffffu, value.row0.z, offset);

		value.row1.x += __shfl_down_sync(0xffffffffu, value.row1.x, offset);
		value.row1.y += __shfl_down_sync(0xffffffffu, value.row1.y, offset);
		value.row1.z += __shfl_down_sync(0xffffffffu, value.row1.z, offset);

		value.row2.x += __shfl_down_sync(0xffffffffu, value.row2.x, offset);
		value.row2.y += __shfl_down_sync(0xffffffffu, value.row2.y, offset);
		value.row2.z += __shfl_down_sync(0xffffffffu, value.row2.z, offset);
	}

	return value;
}
__device__ Float3x3 blockReduceSum(Float3x3 value, Float3x3* warpSums) {
	int laneIndex = threadIdx.x & (warpSize - 1);
	int warpIndex = threadIdx.x / warpSize;
	int warpCount = (blockDim.x + warpSize - 1) / warpSize;

	value = warpReduceSum(value);

	if (laneIndex == 0) {
		warpSums[warpIndex] = value;
	}

	__syncthreads();

	Float3x3 blockSum = makeZeroFloat3x3();

	if (warpIndex == 0) {
		if (laneIndex < warpCount) {
			blockSum = warpSums[laneIndex];
		}

		blockSum = warpReduceSum(blockSum);
	}

	__syncthreads();

	return blockSum;
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

__global__ void updateVerticesKernel(VertexDevice ver, float tstep) {
	int verIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (verIndex >= ver.N) return;
	if (ver.invM[verIndex] == 0.0f) return;

	float3 delta = sub(ver.p[verIndex], ver.pos[verIndex]);
	ver.v[verIndex] = mul(delta, 1.0f / tstep);
	ver.pos[verIndex] = ver.p[verIndex];
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
	bool touched = false;

	float frictionScale = fmaxf(0.0f, 1.0f - clamp(friction, 0.0f, 1.0f));
	float restitutionClamped = clamp(restitution, 0.0f, 1.0f);

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
		touched = true;
		float3 relVn = mul(n, relVnMag);
		float3 relVt = sub(relV, relVn);
		float newRelVnMag = -relVnMag * restitutionClamped;
		float3 newRelVn = mul(n, newRelVnMag);
		float3 newRelVt = mul(relVt, frictionScale);

		float3 newRelV = add(newRelVn, newRelVt);
		v = add(colliderV, newRelV);
	}

	if (touched) {
		ver.v[verIndex] = v;
	}
}
