use afterburner_core::prelude::*;
use afterburner_rustgpu_shared::{RustGpuConv2DParams, RustGpuShape};

use crate::{run_with_backend, RustGpu};

impl Conv2DImpl<RustGpu, f32> for RustGpu {
    fn conv_2d(
        tensor: &Tensor<RustGpu, 4, f32>,
        weights: &Tensor<RustGpu, 4, f32>,
        params: Conv2DParams,
    ) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let stride = params.stride;

            let new_shape: Shape<4> = [
                tensor.shape().as_slice()[0],
                weights.shape().as_slice()[1],
                tensor.shape().as_slice()[2] / stride.as_slice()[0]
                    - weights.shape().as_slice()[2] / 2,
                tensor.shape().as_slice()[3] / stride.as_slice()[1]
                    - weights.shape().as_slice()[3] / 2,
            ]
            .into();

            let output = Tensor::create(new_shape);

            backend.create_buffer(output.id, output.size()).unwrap();

            backend.run_shader(
                "conv2d",
                tensor.id,
                weights.id,
                output.id,
                RustGpuConv2DParams {
                    dimensions: RustGpuShape(tensor.shape.0),
                    conv: RustGpuShape(weights.shape.0),
                    stride: RustGpuShape(stride.0),
                },
            );

            output
        })
    }
}
