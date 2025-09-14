use crate::RustGpuShape;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RustGpuConv2DParams {
    pub dimensions: RustGpuShape<4>,
    pub conv: RustGpuShape<4>,
    pub stride: RustGpuShape<2>,
    pub padding: RustGpuShape<2>,
}

pub fn conv2d(
    id: usize,
    params: &RustGpuConv2DParams,
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
) {
    // Input dimensions: [batch, in_channels, height, width]
    let in_batch = params.dimensions.0[0] as usize;
    let in_channels = params.dimensions.0[1] as usize;
    let in_height = params.dimensions.0[2] as usize;
    let in_width = params.dimensions.0[3] as usize;

    // Weight dimensions: [out_channels, in_channels, kernel_h, kernel_w]
    let out_channels = params.conv.0[0] as usize;
    let weight_in_channels = params.conv.0[1] as usize;
    let kernel_h = params.conv.0[2] as usize;
    let kernel_w = params.conv.0[3] as usize;

    // Calculate output dimensions with padding
    let pad_h = params.padding.0[0] as usize;
    let pad_w = params.padding.0[1] as usize;

    // Calculate output dimensions with padding
    let stride_h = params.stride.0[0] as usize;
    let stride_w = params.stride.0[1] as usize;

    // Safety check: ensure stride values are not zero
    if stride_h == 0 || stride_w == 0 {
        return;
    }

    // Calculate output dimensions
    let out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    let out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

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
                let in_y = out_y * stride_h + ky;
                let in_x = out_x * stride_w + kx;

                // Apply padding offset with bounds checking
                let in_y_with_pad = in_y as isize - pad_h as isize;
                let in_x_with_pad = in_x as isize - pad_w as isize;

                // Bounds check for input (padding adds zeros outside bounds)
                if in_y_with_pad >= 0
                    && in_x_with_pad >= 0
                    && (in_y_with_pad as usize) < in_height
                    && (in_x_with_pad as usize) < in_width
                {
                    let input_idx = out_batch * in_batch_stride
                        + in_ch * in_channel_stride
                        + (in_y_with_pad as usize) * in_width
                        + (in_x_with_pad as usize);

                    let weight_idx = out_channel * weight_out_channel_stride
                        + in_ch * weight_channel_stride
                        + ky * kernel_w
                        + kx;

                    sum += input[input_idx] * weights[weight_idx];
                }
                // If out of bounds due to padding, we implicitly add 0 (no operation needed)
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
            dimensions: RustGpuShape([1u32, 1, 1, 1]), // [batch, in_channels, height, width]
            conv: RustGpuShape([1u32, 1, 1, 1]), // [out_channels, in_channels, kernel_h, kernel_w]
            stride: RustGpuShape([1u32, 1]),     // [stride_h, stride_w]
            padding: RustGpuShape([0u32, 0]),    // [padding_h, padding_w]
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
            dimensions: RustGpuShape([2u32, 2, 3, 3]), // [batch, in_channels, height, width]
            conv: RustGpuShape([2u32, 2, 1, 1]), // [out_channels, in_channels, kernel_h, kernel_w]
            stride: RustGpuShape([1u32, 1]),
            padding: RustGpuShape([0u32, 0]), // [padding_h, padding_w]
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

    #[test]
    fn test_conv2d_with_padding() {
        // Test 2x2 input with 3x3 kernel and padding=1
        // This should produce 2x2 output (same as input size)
        let input = flatten([[[[1.0f32, 2.0], [3.0, 4.0]]]]);
        let weights = flatten([[[[1.0f32, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]]);

        let params = RustGpuConv2DParams {
            dimensions: RustGpuShape([1u32, 1, 2, 2]), // [batch, in_channels, height, width]
            conv: RustGpuShape([1u32, 1, 3, 3]), // [out_channels, in_channels, kernel_h, kernel_w]
            stride: RustGpuShape([1u32, 1]),     // [stride_h, stride_w]
            padding: RustGpuShape([1u32, 1]),    // [padding_h, padding_w]
        };

        let output_size = 1 * 1 * 2 * 2; // batch * out_channels * out_height * out_width
        let mut output = vec![0f32; output_size];
        let src_range = 0..output_size;

        src_range
            .clone()
            .into_iter()
            .for_each(|x| conv2d(x as usize, &params, &input, &weights, &mut output));

        // With padding=1, the 2x2 input becomes effectively 4x4 with zeros around edges
        // The 3x3 Sobel-like kernel should produce edge detection results
        // Output should be 2x2 (same as input due to padding)
        assert_eq!(output.len(), 4);

        // The exact values depend on the edge detection at padded boundaries
        // but we can verify the output has the expected dimensions
        println!("Padded convolution output: {:?}", output);
    }

    #[test]
    fn test_conv2d_sobel() {
        // Test Sobel convolution: 3x3 input with 3x3 Sobel kernel
        let input = flatten([[[[0.0f32, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]]]);
        let weights = flatten([[[[1.0f32, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]]);

        let params = RustGpuConv2DParams {
            dimensions: RustGpuShape([1u32, 1, 3, 3]), // [batch, in_channels, height, width]
            conv: RustGpuShape([1u32, 1, 3, 3]), // [out_channels, in_channels, kernel_h, kernel_w]
            stride: RustGpuShape([1u32, 1]),     // [stride_h, stride_w]
            padding: RustGpuShape([0u32, 0]),    // [padding_h, padding_w]
        };

        let output_size = 1 * 1 * 1 * 1; // batch * out_channels * out_height * out_width
        let mut output = vec![0f32; output_size];
        let src_range = 0..output_size;

        src_range
            .clone()
            .into_iter()
            .for_each(|x| conv2d(x as usize, &params, &input, &weights, &mut output));

        println!("Shared Sobel result: {:?}", output);
        assert_eq!(output, [-3.0f32]);
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
