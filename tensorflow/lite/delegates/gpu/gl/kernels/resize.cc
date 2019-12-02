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

#include "tensorflow/lite/delegates/gpu/gl/kernels/resize.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ResizeNearestNeighbor : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto output = ctx.graph->FindOutputs(ctx.node->id)[0];
    auto input = ctx.graph->FindInputs(ctx.node->id)[0];

    auto h_scale = output->tensor.shape.h / input->tensor.shape.h;
    auto w_scale = output->tensor.shape.w / input->tensor.shape.w;
    auto c_scale = output->tensor.shape.c / input->tensor.shape.c;
    auto shape = input->tensor.shape;

    std::string code = R"(
      int h_offset = gid.x * $h_scale$;
      int w_offset = gid.y * $w_scale$;
      int c_offset = gid.z * $c_scale$;
      for (int h = 0; h < $h_scale$; ++h) {
        for (int w = 0; w < $w_scale$; ++w) {
          for (int c = 0; c < $c_scale$; ++c) {
            vec4 val = $input_data_0[gid.x,gid.y,gid.z]$;
            $output_data_0[h_offset+h,w_offset+w,c_offset+c] = val$;
          }
        }
      }
      )";
    
    std::vector<Variable> parameters = {
      {"h_scale", h_scale},
      {"w_scale", w_scale},
      {"c_scale", c_scale}
    };

    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(shape.w, shape.h, shape.c),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewResizeNearestNeighborNodeShader() {
  return absl::make_unique<ResizeNearestNeighbor>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
