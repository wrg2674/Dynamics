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

void Constraint::GS_Iteration() {
	for (int i = 0; i < vertices.size(); i++){
		if (!satisfyConstraintFunction()) {
			Vertex* cur = vertices.at(i);
			

		}

	}
}

void Constraint::calcCentralDiff(Vertex* cur, float tstep, float result[]) {
	float advP[3], prevP[3], curP[3];
	float advF, prevF = 0;
	float centralDiff[3];

	for (int j = 0; j < 3; j++) {
		// 편미분은 각 성분에 대해 독립적으로 적용되어야 하므로 기존에 변경한 내용을 초기화시켜야함
		for (int i = 0; i < 3; i++) {
			advP[i] = cur->getP()[i];
			prevP[i] = cur->getP()[i];
		}
		advP[j] = cur->getP()[j] + tstep;
		prevP[j] = cur->getP()[j] - tstep;
		curP[j] = cur->getP()[j];

		cur->updateP(advP);
		advF = constraintFunction();
		cur->updateP(prevP);
		prevF = constraintFunction();
		cur->updateP(curP);

		result[j] = (advF - prevF) / (2 * tstep);
	}
}
vector<float[3]> Constraint::calcGradient(float tstep) {
	vector<float[3]> gradient;
	for (int i = 0; i < vertices.size(); i++) {
		Vertex* cur = vertices.at(i);
		float gradientElement[3] = {0};
		calcCentralDiff(cur, tstep, gradientElement);	
		gradient.push_back(gradientElement);
	}
	return gradient;
}
void Constraint::calcDeltaP() {
	for (int i = 0; i < vertices.size(); i++) {

	}
}
void Constraint::projectionFunction() {
	GS_Iteration();
	
}
