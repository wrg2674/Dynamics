#pragma once

#include "Constraint.h"
#include "CudaUtils.cuh"

void uploadColorBatch(const ColorBatchHost& host,ColorBatchDevice& device);