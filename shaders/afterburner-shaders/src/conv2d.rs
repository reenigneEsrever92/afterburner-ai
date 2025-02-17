use afterburner_rustgpu_shared::*;
use spirv_std::spirv;

pub fn conv2d(
    id: usize,
    params: &RustGpuConv2DParams,
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
) {
    let channel_size = params.dimensions.0[2] * params.dimensions.0[3];
    let batch_size = params.dimensions.0[1] * channel_size;

    let batch = id / batch_size;
    let channel = id % batch_size / channel_size;
    let output_y = id % (batch_size + channel_size) / params.dimensions.0[3];
    let output_x = id % (batch_size + channel_size + params.dimensions.0[3]);

    let weights_channel_size = params.conv.as_slice()[2] * params.conv.as_slice()[3];
    let weights_batch_size = params.conv.as_slice()[1] * weights_channel_size;

    for y in 0..params.conv.as_slice()[2] {
        for x in 0..params.conv.as_slice()[3] {
            let input_offset = batch * batch_size
                + channel * channel_size
                + y * params.dimensions.as_slice()[3]
                + x;

            let weight_offset = batch * weights_batch_size
                + channel * weights_channel_size
                + y * params.conv.as_slice()[3]
                + x;

            output[id] += input[input_offset] * weights[weight_offset];
        }
    }
}

#[cfg(test)]
mod test {
    use afterburner_rustgpu_shared::{RustGpuConv2DParams, RustGpuShape};
    use spirv_std::glam::UVec3;
    use tracing_test::traced_test;

    use crate::conv2d::conv2d;

    #[test]
    fn test_conv2d() {
        let input = [1.0];
        let weights = [2.0];
        let mut output = [0.0f32];

        let params = RustGpuConv2DParams {
            dimensions: RustGpuShape([1, 1, 1, 1]),
            conv: RustGpuShape([1, 1, 1, 1]),
            stride: RustGpuShape([1, 1]),
        };

        let src_range = 0..1;

        let result = src_range
            .clone()
            .into_iter()
            .for_each(|x| conv2d(x as usize, &params, &input, &weights, &mut output));

        assert_eq!(output, [2.0f32]);
    }
}
