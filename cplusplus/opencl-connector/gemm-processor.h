#pragma once

#include "opencl-connector.h"

namespace NOpenCLConnector {

class GEMMProcessor {
public:
    GEMMProcessor(OpenCLConnector& opencl_connector);
private:
    OpenCLConnector&     opencl_connector;
    cl::Program::Sources sources;
    cl::Program          program;
};

} // namespace NOpenCLConnector
