#include "CommonSolver.cuh"

__device__ void calcCentralDiff(VertexDevice ver, ConstraintDevice cons, int verIndex, int consIndex, float tstep, Type type, float3& result) {
	//float tstep = 1e-4; // tstep은 너무커서 더 작은 값으로 중심차분법을 계산함
	if (tstep <= 0.0f) { result = make_float3(0, 0, 0); return; }
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
		set(advP, j, get(curP, j) + tstep);
		set(prevP, j, get(curP, j) - tstep);

		switch (type) {
		case Stretch:
			advF = calcStretchOverride(ver, cons, consIndex, verIndex, advP);
			prevF = calcStretchOverride(ver, cons, consIndex, verIndex, prevP);
			break;
		case Bending:
			advF = calcBendingOverride(ver, cons, consIndex, verIndex, advP);
			prevF = calcBendingOverride(ver, cons, consIndex, verIndex, prevP);
		}
		float temp = (advF-prevF)/ (2 * tstep);
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

	cross(p1_0, p2_0, firstTerm);
	float firstLength = length(firstTerm);
	if (firstLength < 1e-8f) {
		return 0;
	}
	firstTerm = normalize(firstTerm);

	cross(p1_0, p3_0, secondTerm);
	float secondLength = length(secondTerm);
	if (secondLength < 1e-8f) {
		return 0;
	}
	secondTerm = normalize(secondTerm);


	float dotTerm = dot(firstTerm, secondTerm);
	dotTerm = clamp(dotTerm, -1.0f, 1.0f); //부동 소수점 오차로 인한 오류 방지
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
		cross(r, velocity, tmp);
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

	float3 n;
	cross(e1, e2, n);

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
__global__ void clearCollisionFlags(bool* collided) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < gridDim.x * blockDim.x) {
		collided[i] = false;
	}
}
__global__ void resolvePlaneCollisionKernel(VertexDevice ver, CollisionPlane plane, float3* normals, float* fricOut, float* restOut, bool* collided) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ver.N || ver.invM[i] == 0.0f) {
		return;
	}
	float dist = dot(ver.p[i], plane.n) - plane.offset;
	if (dist < 0.0f) {
		ver.p[i] = add(ver.p[i], mul(plane.n, -dist));
		normals[i] = plane.n;
		fricOut[i] = plane.friction;
		restOut[i] = plane.restitution;
		collided[i] = true;
	}
}
__global__ void resolveSphereCollisionKernel(VertexDevice ver, CollisionSphere sphere, float3* normals, float* fricOut, float* restOut, bool* collided) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ver.N || ver.invM[i] == 0.0f) {
		return;
	}
	float3 d = sub(ver.p[i], sphere.c);
	float len = length(d);
	float pen = sphere.r - len;
	if (pen > 0.0f && len > 1e-8f) {
		float3 n;
		n = normalize(d);
		ver.p[i] = add(ver.p[i], mul(n, pen));
		normals[i] = n;
		fricOut[i] = sphere.friction;
		restOut[i] = sphere.restitution;
		collided[i] = true;
	}
}
__global__ void applyCollisionVelocityKernel(VertexDevice ver, const float3* normals, const float* fricOut, const float* restOut, const bool* collided) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= ver.N || ver.invM[i] == 0.0f) {
		return;
	}
	if (!collided[i]) {
		return;
	}
	float3 n = normals[i];
	float vn = dot(ver.v[i], n);
	float3 vt = sub(ver.v[i], mul(n, vn));

	if (vn < 0) {
		vn = -vn*restOut[i];
	}
	float3 vNormal = mul(n, vn);
	float3 vTangent = mul(vt, fmaxf(0.0f, 1.0f - fricOut[i]));
	ver.v[i] = add(vNormal, vTangent);
}