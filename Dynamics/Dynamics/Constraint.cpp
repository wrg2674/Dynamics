#include "Constraint.h"

Constraint::Constraint(){
	this->cardinality = 0;
	this->index = vector<int>();
	this->k = 0;
	this->type = true;
}

Constraint::Constraint(int cardinality, float k, bool type) {
	this->cardinality = cardinality;
	this->index = vector<int>();
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
		return constraintFunction() == 0;
	}
	return constraintFunction() < 0;
}

void Constraint::GS_Iteration(float tstep, int ns) {
	for (int i = 0; i < vertices.size(); i++){
		if (!satisfyConstraintFunction()) {
			Vertex* cur = vertices.at(i);
			vector<glm::vec3> gradient;
			gradient = calcGradient(tstep);
			calcDeltaP(i,gradient,tstep);
			for (int j = 0; j < 3; j++) {
				cur->p[j] = cur->p[j] + (1.0-pow((1.0-k), 1.0/ns)) * cur->dp[j];
			}
		}

	}
}

void Constraint::calcCentralDiff(Vertex* cur, float tstep, glm::vec3& result) {
	glm::vec3 advP, prevP, curP;
	float advF, prevF = 0;
	glm::vec3 centralDiff;

	for (int j = 0; j < 3; j++) {
		// 편미분은 각 성분에 대해 독립적으로 적용되어야 하므로 기존에 변경한 내용을 초기화시켜야함
		for (int i = 0; i < 3; i++) {
			advP[i] = cur->p[i];
			prevP[i] = cur->p[i];
		}
		advP[j] = cur->p[j] + tstep;
		prevP[j] = cur->p[j] - tstep;
		curP[j] = cur->p[j];

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
		sumGradientNorm += (1.0 / vertices.at(j)->m) * (gradient.at(j).length()*gradient.at(j).length());
	}
	float s = constraintFunction() / sumGradientNorm;
	for (int j = 0; j < 3; j++) {
		vertices.at(idx)->dp[j] = -s * (1.0 / vertices.at(idx)->m) * gradient[idx][j];
	}
}
void Constraint::projectionFunction(float tstep, int ns) {
	GS_Iteration(tstep, ns);
}
