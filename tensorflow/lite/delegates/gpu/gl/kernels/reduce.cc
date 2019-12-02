/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/gl/kernels/reduce.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ReduceMax : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    const auto* input = ctx.graph->FindInputs(ctx.node->id)[0];
    const auto* output = ctx.graph->FindOutputs(ctx.node->id)[0];

    // TODO: check specified axis, see Softmax code

    std::vector<Variable> parameters = {
        {"src_depth", IntegralDivideRoundUp(input->tensor.shape.c, 4)},
    };

    std::string code = R"(
      float max = 0.0f;
      for (int d = 0; d < $src_depth$; ++d) {
        float val = float($input_data_0[gid.x, gid.y, d]$);
        if (val > max)
            max = val;
      }
      $output_data_0[gid.x, gid.y, 0] = vec4(max)$;
    )";

    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(output->tensor.shape.w, output->tensor.shape.h, 1),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewReduceMaxNodeShader() {
  return absl::make_unique<ReduceMax>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
