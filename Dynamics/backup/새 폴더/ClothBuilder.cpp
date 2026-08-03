#include "ClothBuilder.h"

#include <unordered_map>
#include <algorithm>
#include <cmath>

#include "Vertex.h"

struct EdgeKey {
	unsigned int a;
	unsigned int b;

	bool operator==(const EdgeKey& o) const {
		return a == o.a && b == o.b;
	}
};

struct EdgeKeyHash {
	size_t operator()(const EdgeKey& k) const {
		return (static_cast<size_t>(k.a) << 32) ^ static_cast<size_t>(k.b);
	}
};

struct EdgeInfo {
	unsigned int a;
	unsigned int b;
	unsigned int opp;
};

static EdgeKey makeKey(unsigned int u, unsigned int v) {
	if (u < v) return { u, v };
	return { v, u };
}

static bool shareVertex(const int2& a, const int2& b) {
	return a.x == b.x || a.x == b.y || a.y == b.x || a.y == b.y;
}

static bool shareVertex(const int4& a, const int4& b) {
	int av[4] = { a.x, a.y, a.z, a.w };
	int bv[4] = { b.x, b.y, b.z, b.w };

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			if (av[i] == bv[j]) {
				return true;
			}
		}
	}

	return false;
}

void buildClothPositions(std::vector<glm::vec3>& h_pos, std::vector<glm::vec3>& h_v, std::vector<glm::vec3>& h_p, std::vector<glm::vec3>& h_dp, std::vector<float>& h_invM, int rows, int cols) {
	const int N = rows * cols;

	h_pos.resize(N);
	h_v.resize(N);
	h_p.resize(N);
	h_dp.resize(N);
	h_invM.resize(N);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int idx = i * cols + j;

			float px = j / static_cast<float>(cols - 1) - (1.0f - j / static_cast<float>(cols - 1));
			float py = -i / static_cast<float>(rows - 1) + (1.0f - i / static_cast<float>(rows - 1));
			float pz = 0.0f;

			h_pos[idx] = glm::vec3(px, py, pz);
			h_v[idx] = glm::vec3(0.0f);
			h_p[idx] = h_pos[idx];
			h_dp[idx] = glm::vec3(0.0f);
			h_invM[idx] = 1.0f;
		}
	}

	h_invM[0] = 0.0f;
	h_invM[cols - 1] = 0.0f;
}

void buildTriangleIndices(std::vector<unsigned int>& indices, int rows, int cols) {
	indices.clear();

	for (int i = 0; i < rows - 1; i++) {
		for (int j = 0; j < cols - 1; j++) {
			unsigned int i0 = i * cols + j;
			unsigned int i1 = i * cols + j + 1;
			unsigned int i2 = (i + 1) * cols + j;
			unsigned int i3 = (i + 1) * cols + j + 1;

			indices.push_back(i0);
			indices.push_back(i1);
			indices.push_back(i2);

			indices.push_back(i1);
			indices.push_back(i3);
			indices.push_back(i2);
		}
	}
}

void buildStretchConstraints(ConstraintHost& cons, const std::vector<glm::vec3>& h_pos, int rows, int cols, float stretchK) {
	cons.stretch.ver.clear();
	cons.stretch.k.clear();
	cons.stretch.l0.clear();

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols - 1; j++) {
			int a = i * cols + j;
			int b = i * cols + j + 1;

			glm::vec3 pa = h_pos[a];
			glm::vec3 pb = h_pos[b];

			float dx = pb.x - pa.x;
			float dy = pb.y - pa.y;
			float dz = pb.z - pa.z;
			float rest = std::sqrt(dx * dx + dy * dy + dz * dz);

			cons.stretch.ver.push_back(make_int2(a, b));
			cons.stretch.k.push_back(stretchK);
			cons.stretch.l0.push_back(rest);
		}
	}

	for (int i = 0; i < rows - 1; i++) {
		for (int j = 0; j < cols; j++) {
			int a = i * cols + j;
			int b = (i + 1) * cols + j;

			glm::vec3 pa = h_pos[a];
			glm::vec3 pb = h_pos[b];

			float dx = pb.x - pa.x;
			float dy = pb.y - pa.y;
			float dz = pb.z - pa.z;
			float rest = std::sqrt(dx * dx + dy * dy + dz * dz);

			cons.stretch.ver.push_back(make_int2(a, b));
			cons.stretch.k.push_back(stretchK);
			cons.stretch.l0.push_back(rest);
		}
	}

	for (int i = 0; i < rows - 1; i++) {
		for (int j = 0; j < cols - 1; j++) {
			int p1 = i * cols + j;
			int p2 = (i + 1) * cols + j + 1;
			int p3 = i * cols + j + 1;
			int p4 = (i + 1) * cols + j;

			glm::vec3 pa = h_pos[p1];
			glm::vec3 pb = h_pos[p2];

			float dx = pb.x - pa.x;
			float dy = pb.y - pa.y;
			float dz = pb.z - pa.z;
			float rest = std::sqrt(dx * dx + dy * dy + dz * dz);

			cons.stretch.ver.push_back(make_int2(p1, p2));
			cons.stretch.k.push_back(stretchK);
			cons.stretch.l0.push_back(rest);

			pa = h_pos[p3];
			pb = h_pos[p4];

			dx = pb.x - pa.x;
			dy = pb.y - pa.y;
			dz = pb.z - pa.z;
			rest = std::sqrt(dx * dx + dy * dy + dz * dz);

			cons.stretch.ver.push_back(make_int2(p3, p4));
			cons.stretch.k.push_back(stretchK);
			cons.stretch.l0.push_back(rest);
		}
	}
}

void buildBendingConstraints_AllSharedEdges(const std::vector<unsigned int>& indices, const std::vector<glm::vec3>& h_pos, ConstraintHost& cons, float bendingK) {
	cons.bending.ver.clear();
	cons.bending.k.clear();
	cons.bending.phi0.clear();

	std::unordered_map<EdgeKey, EdgeInfo, EdgeKeyHash> edgeMap;

	for (size_t t = 0; t < indices.size(); t += 3) {
		unsigned int tri[3] = { indices[t], indices[t + 1], indices[t + 2] };

		for (int e = 0; e < 3; e++) {
			unsigned int a = tri[e];
			unsigned int b = tri[(e + 1) % 3];
			unsigned int opp = tri[(e + 2) % 3];

			EdgeKey key = makeKey(a, b);
			auto it = edgeMap.find(key);

			if (it == edgeMap.end()) {
				edgeMap[key] = { a, b, opp };
			}
			else {
				unsigned int v0 = it->second.a;
				unsigned int v1 = it->second.b;
				unsigned int v2 = it->second.opp;
				unsigned int v3 = opp;

				glm::vec3 p0 = h_pos[v0];
				glm::vec3 p1 = h_pos[v1];
				glm::vec3 p2 = h_pos[v2];
				glm::vec3 p3 = h_pos[v3];

				glm::vec3 e01 = p1 - p0;
				float edgeLen = glm::length(e01);
				float phi0 = 0.0f;

				if (edgeLen > 1e-8f) {
					glm::vec3 n1 = glm::cross(p2 - p0, e01);
					glm::vec3 n2 = glm::cross(e01, p3 - p0);

					float n1Len = glm::length(n1);
					float n2Len = glm::length(n2);

					if (n1Len > 1e-8f && n2Len > 1e-8f) {
						float cval = glm::dot(n1, n2) / (n1Len * n2Len);
						cval = glm::clamp(cval, -1.0f, 1.0f);
						phi0 = glm::acos(cval);

						if (glm::dot(glm::cross(n1, n2), e01) > 0.0f) {
							phi0 = -phi0;
						}
					}
				}
				

				cons.bending.ver.push_back(make_int4(v0, v1, v2, v3));
				cons.bending.k.push_back(bendingK);
				cons.bending.phi0.push_back(phi0);
			}
		}
	}
}

void buildPerVertexConstraintLists(const ConstraintHost& cons, std::vector<unsigned int>& constraintsArray, std::vector<int>& constraintOffset, int vertexCount) {
	std::vector<std::vector<unsigned int>> perVertex(vertexCount);

	for (int i = 0; i < static_cast<int>(cons.stretch.ver.size()); i++) {
		int2 edge = cons.stretch.ver[i];

		perVertex[edge.x].push_back(packCons(Stretch, static_cast<unsigned int>(i)));
		perVertex[edge.y].push_back(packCons(Stretch, static_cast<unsigned int>(i)));
	}

	for (int i = 0; i < static_cast<int>(cons.bending.ver.size()); i++) {
		int4 bend = cons.bending.ver[i];

		perVertex[bend.x].push_back(packCons(Bending, static_cast<unsigned int>(i)));
		perVertex[bend.y].push_back(packCons(Bending, static_cast<unsigned int>(i)));
		perVertex[bend.z].push_back(packCons(Bending, static_cast<unsigned int>(i)));
		perVertex[bend.w].push_back(packCons(Bending, static_cast<unsigned int>(i)));
	}

	constraintsArray.clear();
	constraintOffset.assign(vertexCount + 1, 0);

	int running = 0;

	for (int i = 0; i < vertexCount; i++) {
		constraintOffset[i] = running;
		running += static_cast<int>(perVertex[i].size());
	}

	constraintOffset[vertexCount] = running;

	for (int i = 0; i < vertexCount; i++) {
		for (unsigned int packed : perVertex[i]) {
			constraintsArray.push_back(packed);
		}
	}
}

void buildStretchColorBatches(ConstraintHost& cons) {
	int n = static_cast<int>(cons.stretch.ver.size());
	std::vector<std::vector<int>> groups;

	for (int i = 0; i < n; i++) {
		int chosenColor = -1;

		for (int c = 0; c < static_cast<int>(groups.size()); c++) {
			bool conflict = false;

			for (int id : groups[c]) {
				if (shareVertex(cons.stretch.ver[i], cons.stretch.ver[id])) {
					conflict = true;
					break;
				}
			}

			if (!conflict) {
				chosenColor = c;
				break;
			}
		}

		if (chosenColor == -1) {
			groups.push_back({});
			chosenColor = static_cast<int>(groups.size()) - 1;
		}

		groups[chosenColor].push_back(i);
	}

	cons.stretch.color.constraintIds.clear();
	cons.stretch.color.colorOffset.clear();
	cons.stretch.color.colorOffset.push_back(0);

	for (int c = 0; c < static_cast<int>(groups.size()); c++) {
		for (int id : groups[c]) {
			cons.stretch.color.constraintIds.push_back(id);
		}

		cons.stretch.color.colorOffset.push_back(static_cast<int>(cons.stretch.color.constraintIds.size()));
	}
}

void buildBendingColorBatches(ConstraintHost& cons) {
	int n = static_cast<int>(cons.bending.ver.size());
	std::vector<std::vector<int>> groups;

	for (int i = 0; i < n; i++) {
		int chosenColor = -1;

		for (int c = 0; c < static_cast<int>(groups.size()); c++) {
			bool conflict = false;

			for (int id : groups[c]) {
				if (shareVertex(cons.bending.ver[i], cons.bending.ver[id])) {
					conflict = true;
					break;
				}
			}

			if (!conflict) {
				chosenColor = c;
				break;
			}
		}

		if (chosenColor == -1) {
			groups.push_back({});
			chosenColor = static_cast<int>(groups.size()) - 1;
		}

		groups[chosenColor].push_back(i);
	}

	cons.bending.color.constraintIds.clear();
	cons.bending.color.colorOffset.clear();
	cons.bending.color.colorOffset.push_back(0);

	for (int c = 0; c < static_cast<int>(groups.size()); c++) {
		for (int id : groups[c]) {
			cons.bending.color.constraintIds.push_back(id);
		}

		cons.bending.color.colorOffset.push_back(static_cast<int>(cons.bending.color.constraintIds.size()));
	}
}

void buildVertexTriangleAdjacency(const std::vector<unsigned int>& triangleIndices, int vertexCount, std::vector<int>& outTriArray, std::vector<int>& outTriOffset) {
	std::vector<std::vector<int>> perVertex(vertexCount);
	int triCount = static_cast<int>(triangleIndices.size()) / 3;

	for (int t = 0; t < triCount; t++) {
		int i0 = static_cast<int>(triangleIndices[3 * t + 0]);
		int i1 = static_cast<int>(triangleIndices[3 * t + 1]);
		int i2 = static_cast<int>(triangleIndices[3 * t + 2]);

		perVertex[i0].push_back(t);
		perVertex[i1].push_back(t);
		perVertex[i2].push_back(t);
	}

	outTriArray.clear();
	outTriOffset.assign(vertexCount + 1, 0);

	int running = 0;

	for (int v = 0; v < vertexCount; v++) {
		outTriOffset[v] = running;
		running += static_cast<int>(perVertex[v].size());
	}

	outTriOffset[vertexCount] = running;

	outTriArray.reserve(running);

	for (int v = 0; v < vertexCount; v++) {
		for (int triIdx : perVertex[v]) {
			outTriArray.push_back(triIdx);
		}
	}
}