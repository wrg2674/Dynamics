#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "Constraint.h"

void buildClothPositions(std::vector<glm::vec3>& h_pos, std::vector<glm::vec3>& h_v, std::vector<glm::vec3>& h_p, std::vector<glm::vec3>& h_dp, std::vector<float>& h_invM, int rows, int cols);
void buildTriangleIndices(std::vector<unsigned int>& indices, int rows, int cols);
void buildStretchConstraints(ConstraintHost& cons, const std::vector<glm::vec3>& h_pos, int rows, int cols, float stretchK);
void buildBendingConstraints_AllSharedEdges(const std::vector<unsigned int>& indices, const std::vector<glm::vec3>& h_pos, ConstraintHost& cons, float bendingK);
void buildPerVertexConstraintLists(const ConstraintHost& cons, std::vector<unsigned int>& constraintsArray, std::vector<int>& constraintOffset, int vertexCount);
void buildStretchColorBatches(ConstraintHost& cons);
void buildBendingColorBatches(ConstraintHost& cons);
void buildVertexTriangleAdjacency(const std::vector<unsigned int>& triangleIndices, int vertexCount, std::vector<int>& outTriArray, std::vector<int>& outTriOffset);