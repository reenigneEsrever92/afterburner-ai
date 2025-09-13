use crate::RustGpuShape;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RustGpuConv2DParams {
    pub dimensions: RustGpuShape<4>,
    pub conv: RustGpuShape<4>,
    pub stride: RustGpuShape<2>,
}

pub fn conv2d(
    id: usize,
    params: &RustGpuConv2DParams,
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
) {
    // Input dimensions: [batch, in_channels, height, width]
    let in_batch = params.dimensions.0[0];
    let in_channels = params.dimensions.0[1];
    let in_height = params.dimensions.0[2];
    let in_width = params.dimensions.0[3];

    // Weight dimensions: [out_channels, in_channels, kernel_h, kernel_w]
    let out_channels = params.conv.0[0];
    let weight_in_channels = params.conv.0[1];
    let kernel_h = params.conv.0[2];
    let kernel_w = params.conv.0[3];

    // Calculate output dimensions
    let out_height = (in_height - kernel_h) / params.stride.0[0] + 1;
    let out_width = (in_width - kernel_w) / params.stride.0[1] + 1;

    // Calculate strides for indexing
    let in_channel_stride = in_height * in_width;
    let in_batch_stride = in_channels * in_channel_stride;

    let out_channel_stride = out_height * out_width;
    let out_batch_stride = out_channels * out_channel_stride;

    let weight_channel_stride = kernel_h * kernel_w;
    let weight_out_channel_stride = weight_in_channels * weight_channel_stride;

    // Decode output position from linear id
    let out_batch = id / out_batch_stride;
    let out_channel = (id % out_batch_stride) / out_channel_stride;
    let out_y = ((id % out_batch_stride) % out_channel_stride) / out_width;
    let out_x = ((id % out_batch_stride) % out_channel_stride) % out_width;

    // Bounds check
    if out_batch >= in_batch
        || out_channel >= out_channels
        || out_y >= out_height
        || out_x >= out_width
    {
        return;
    }

    let mut sum = 0.0f32;

    // Convolution operation
    for in_ch in 0..weight_in_channels {
        for ky in 0..kernel_h {
            for kx in 0..kernel_w {
                let in_y = out_y * params.stride.0[0] + ky;
                let in_x = out_x * params.stride.0[1] + kx;

                // Bounds check for input
                if in_y < in_height && in_x < in_width {
                    let input_idx = out_batch * in_batch_stride
                        + in_ch * in_channel_stride
                        + in_y * in_width
                        + in_x;

                    let weight_idx = out_channel * weight_out_channel_stride
                        + in_ch * weight_channel_stride
                        + ky * kernel_w
                        + kx;

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    output[id] = sum;
}

#[cfg(test)]
mod test {
    use core::fmt::Debug;

    use crate::RustGpuShape;

    use crate::conv2d::{conv2d, RustGpuConv2DParams};

    #[test]
    fn test_conv2d() {
        let input = [1.0];
        let weights = [2.0];
        let mut output = [0.0f32];

        let params = RustGpuConv2DParams {
            dimensions: RustGpuShape([1, 1, 1, 1]), // [batch, in_channels, height, width]
            conv: RustGpuShape([1, 1, 1, 1]), // [out_channels, in_channels, kernel_h, kernel_w]
            stride: RustGpuShape([1, 1]),     // [stride_h, stride_w]
        };

        let src_range = 0..1;

        src_range
            .clone()
            .into_iter()
            .for_each(|x| conv2d(x as usize, &params, &input, &weights, &mut output));

        assert_eq!(output, [2.0f32]);
    }

    #[test]
    fn test_conv2d_2() {
        let input = flatten([
            [
                [[1.0f32, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            ],
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            ],
        ]);

        let weights = flatten([[[[2.0f32]], [[3.0]]], [[[4.0]], [[5.0]]]]);

        let params = RustGpuConv2DParams {
            dimensions: RustGpuShape([2, 2, 3, 3]), // [batch, in_channels, height, width]
            conv: RustGpuShape([2, 2, 1, 1]), // [out_channels, in_channels, kernel_h, kernel_w]
            stride: RustGpuShape([1, 1]),
        };

        let output_size = 2 * 2 * 3 * 3; // batch * out_channels * out_height * out_width
        let mut output = vec![0f32; output_size];
        let src_range = 0..output_size;

        src_range
            .clone()
            .into_iter()
            .for_each(|x| conv2d(x as usize, &params, &input, &weights, &mut output));

        let expected = flatten([
            [
                [[8f32, 8., 8.], [8., 8., 8.], [8., 8., 8.]],
                [[14., 14., 14.], [14., 14., 14.], [14., 14., 14.]],
            ],
            [
                [[8., 8., 8.], [8., 8., 8.], [8., 8., 8.]],
                [[14., 14., 14.], [14., 14., 14.], [14., 14., 14.]],
            ],
        ]);

        assert_eq!(output, expected);
    }

    fn flatten<const D: usize, const D2: usize, const D3: usize, const D4: usize, T: Debug>(
        data: [[[[T; D]; D2]; D3]; D4],
    ) -> [T; D * D2 * D3 * D4] {
        data.into_iter()
            .flatten()
            .into_iter()
            .flatten()
            .into_iter()
            .flatten()
            .into_iter()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}
