#include "BendingConstraint.h"

using namespace std;

BendingConstraint::BendingConstraint(int cardinality, float k, bool type, float phi) :Constraint(cardinality, k, type) {
	this->phi = phi;
}
float BendingConstraint::calc() {
	glm::vec3 firstTerm = glm::normalize(glm::cross((vertices.at(1)->pos - vertices.at(0)->pos), (vertices.at(2)->pos - vertices.at(0)->pos)));
	glm::vec3 secondTerm = glm::normalize(glm::cross((vertices.at(1)->pos - vertices.at(0)->pos), (vertices.at(3)->pos - vertices.at(0)->pos)));
	// 삼각형이 직선에 가까워지면 법선/각도가 정의 안되므로 이런 경우 phi 반환
	float firstLength = glm::length(firstTerm);
	float secondLength = glm::length(secondTerm);
	if (firstLength < 1e-8f || secondLength < 1e-8f) {
		return phi;
	}
	float dot = glm::clamp(glm::dot(firstTerm, secondTerm), -1.0f, 1.0f); //부동 소수점 오차로 인한 오류 방지
	return glm::acos(dot);
}
float BendingConstraint::constraintFunction() {
	
	return calc() - phi;
}