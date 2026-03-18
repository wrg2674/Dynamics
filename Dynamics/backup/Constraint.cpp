#include "Constraint.h"


Constraint::Constraint(){
	this->cardinality = 0;
	this->index = vector<int>();
	this->vertices = vector<Vertex*>();
	this->k = 0;
	this->type = true;
}

Constraint::Constraint(int cardinality, float k, bool type) {
	this->cardinality = cardinality;
	this->index = vector<int>();
	this->vertices = vector<Vertex*>();
	this->k = k;
	this->type = type;
}

void Constraint::addIndex(int idx) {
	index.push_back(idx);
}
void Constraint::addVertex(Vertex* item) {
	vertices.push_back(item);
}
bool Constraint::satisfyConstraintFunction() {
	if (type) {
		return abs(constraintFunction()) <= 1e-5f; // 원래는 == 0 이지만, 부동소수점 오차 고려함
	}
	return constraintFunction() < 0;
}

void Constraint::calcCentralDiff(Vertex* cur, float tstep, glm::vec3& result) {
	//float tstep = 1e-4; // tstep은 너무커서 더 작은 값으로 중심차분법을 계산함
	glm::vec3 advP, prevP, curP;
	float advF, prevF = 0;
	glm::vec3 centralDiff;

	for (int i = 0; i < 3; i++) {
		curP[i] = cur->p[i];
	}
	for (int j = 0; j < 3; j++) {
		// 편미분은 각 성분에 대해 독립적으로 적용되어야 하므로 
		// 기존에 변경한 내용을 초기화시켜야함
		for (int i = 0; i < 3; i++) {
			advP[i] = cur->p[i];
			prevP[i] = cur->p[i];
		}
		advP[j] = cur->p[j] + tstep;
		prevP[j] = cur->p[j] - tstep;

		cur->updateP(advP);
		advF = constraintFunction();
		cur->updateP(prevP);
		prevF = constraintFunction();
		cur->updateP(curP);

		result[j] = (advF - prevF) / (2 * tstep);
	}
}
vector<glm::vec3> Constraint::calcGradient(float tstep) {
	vector<glm::vec3> gradient;
	for (int i = 0; i < vertices.size(); i++) {
		Vertex* cur = vertices.at(i);
		glm::vec3 gradientElement = {0, 0, 0};
		calcCentralDiff(cur, tstep, gradientElement);	
		gradient.push_back(gradientElement);
	}
	return gradient;
}
void Constraint::calcDeltaP(int idx , vector<glm::vec3>& gradient,float tstep){
	float sumGradientNorm = 0;
	for (int j = 0; j < gradient.size(); j++) {
		sumGradientNorm += (1.0 / vertices.at(j)->m) * pow(glm::length(gradient.at(j)),2);
	}
	float s = constraintFunction() / sumGradientNorm;
	for (int j = 0; j < 3; j++) {
		vertices.at(idx)->dp[j] = -s * (1.0 / vertices.at(idx)->m) * gradient[idx][j];
	}
}
void Constraint::projectionFunction(float tstep, int ns) {
	if (satisfyConstraintFunction()) {
		return;
	}
	vector<glm::vec3> gradient;
	gradient = calcGradient(tstep);
	for (int i = 0; i < vertices.size(); i++) {
		Vertex* cur = vertices.at(i);
		if (cur->pinned) {
			continue;
		}
		calcDeltaP(i, gradient, tstep);
	}
	float weight = 1.0 - pow((1.0 - k), 1.0 / ns);
	// GS 스타일의 즉시 업데이트는 제약사항 단위의 것을 의미하는 것이지 
	// 한 제약사항 내에서 각 정점마다 즉시 업데이트를 하면 안됨.
	for (int i = 0; i < vertices.size(); i++) {
		for (int j = 0; j < 3; j++) {
			Vertex* cur = vertices.at(i);
			if (cur->pinned) {
				continue;
			}
			cur->p[j] = cur->p[j] +  weight* cur->dp[j];
		}
	}
}