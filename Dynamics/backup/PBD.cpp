#include "PBD.h"

using namespace std;

PBD::PBD(vector<Vertex>& vertices, vector<glm::vec3>& forces, vector<Constraint*>& constraints, float tstep, float k_damping, float iterationCount)
	:vertices(vertices), forces(forces), constraints(constraints),tstep(tstep),k_damping(k_damping),iterationCount(iterationCount)
{

}
void PBD::solve() {
	applyForce();
	dampVelocities();
	estimateP();
	// generateCollisionConstraint();
	GSiteration();
	updateVertices();
	// velocityUpdate();
}

void PBD::sumExtForce(vector<glm::vec3>& result) {
	for (int k = 0; k < vertices.size(); k++) {
		for (int i = 0; i < forces.size(); i++) {
			for (int j = 0; j < 3; j++) {
				result[k][j] += forces.at(i)[j] * vertices.at(k).m;
			}
		}
	}
}
void PBD::applyForce() {
	vector<glm::vec3> vertexForce = vector<glm::vec3>();
	for (int i = 0; i < vertices.size(); i++) {
		vertexForce.push_back({ 0,0,0 });
	}
	sumExtForce(vertexForce);
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			cur.v[j] = cur.v[j] + tstep * (1.0 / cur.m) * vertexForce[i][j];
			if (cur.pinned) {
				cur.v[j] = 0;
			}
		}
	}
}
void PBD::dampVelocities() {
	glm::vec3 poscm = { 0,0,0 };
	glm::vec3 vcm = { 0,0,0 };
	glm::vec3 sumPosm = { 0,0,0 };
	glm::vec3 sumVm = { 0,0,0 };
	float sumM = 0;
	glm::vec3 L = { 0,0,0 }, w = {0,0,0};
	glm::mat3 I=glm::mat3(1.0f);
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			sumPosm[j] += cur.pos[j] * cur.m;
			sumVm[j] += cur.v[j] * cur.m;
			sumM += cur.m;
		}
	}
	for (int j = 0; j < 3; j++) {
		poscm[j] = sumPosm[j] / sumM;
		vcm[j] = sumVm[j] / sumM;
	}
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		glm::vec3 r = { 0,0,0 };
		for (int j = 0; j < 3; j++) {
			r[j] = cur.pos[j] - poscm[j];
		}
		L += glm::cross(r, cur.m * cur.v);
		glm::mat3 skew = { {0, r[2], -r[1]}, {-r[2], 0, r[0]}, {r[1], -r[0], 0} };
		I += skew * glm::transpose(skew) * cur.m;
	}
	w = glm::inverse(I) * L;
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		glm::vec3 deltaV = vcm + glm::cross(w, cur.pos - poscm) - cur.v;
		cur.v = cur.v + k_damping * deltaV;
	}
}

void PBD::estimateP() {
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		cur.p = cur.pos + cur.v * tstep;
		if (cur.pinned) {
			cur.p = cur.pos;
		}
	}
}
CollisionDetection PBD::CCD() {
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		glm::vec3 ray = { 0,0,0 };
		for (int j = 0; j < 3; j++) {
			ray[j] = cur.p[j] - cur.pos[j];
		}

	}
	return FAIL;
}
void PBD::generateCollisionConstraint() {

}

void PBD::GSiteration() {
	// GS 스타일의 즉시 업데이트는 제약사항 단위의 것을 의미하는 것이지 
	// 한 제약사항 내에서 각 정점마다 즉시 업데이트를 하면 안됨.
	for (int count = 0; count < iterationCount; count++) {
		for (int i = 0; i < constraints.size(); i++) {
			constraints.at(i)->projectionFunction(tstep, iterationCount);
		}
	}

}
void PBD::updateVertices() {
	for (int i = 0; i < vertices.size(); i++) {
		Vertex& cur = vertices.at(i);
		for (int j = 0; j < 3; j++) {
			cur.v[j] = (cur.p[j] - cur.pos[j]) / tstep;
			if (cur.pinned) {
				cur.v[j] = 0.0f;
			}
		}
		for (int j = 0; j < 3; j++) {
			cur.pos[j] = cur.p[j];
			if (cur.pinned) {
				cur.pos[j] = cur.pos[j];
			}
		}
	}
}
void PBD::velocityUpdate() {

}
