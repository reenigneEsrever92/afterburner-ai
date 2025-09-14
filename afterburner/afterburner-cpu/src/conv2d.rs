use tracing::debug;

use crate::prelude::*;

impl Conv2DImpl<Cpu, f32> for Cpu {
    fn conv_2d(
        tensor: &Tensor<Cpu, 4, f32>,
        weights: &Tensor<Cpu, 4, f32>,
        params: Conv2DParams,
    ) -> Tensor<Cpu, 4, f32> {
        debug!(?tensor, ?weights, ?params, "Convoluting tensor");

        let stride = &params.stride;
        let padding = &params.padding;

        // Calculate output dimensions with padding:
        // out_size = (input_size + 2*padding - kernel_size) / stride + 1
        let in_height = tensor.shape().as_slice()[2];
        let in_width = tensor.shape().as_slice()[3];
        let kernel_height = weights.shape().as_slice()[2];
        let kernel_width = weights.shape().as_slice()[3];

        let out_height =
            (in_height + 2 * padding.as_slice()[0] - kernel_height) / stride.as_slice()[0] + 1;
        let out_width =
            (in_width + 2 * padding.as_slice()[1] - kernel_width) / stride.as_slice()[1] + 1;

        let new_shape: Shape<4> = [
            tensor.shape().as_slice()[0],  // batch size
            weights.shape().as_slice()[0], // output channels
            out_height,
            out_width,
        ]
        .into();

        // new tensor
        let mut result: Vec<f32> = vec![0f32; new_shape.size()];

        let input_data = tensor.to_vec();
        let weight_data = weights.to_vec();
        let pad_h = padding.as_slice()[0];
        let pad_w = padding.as_slice()[1];

        for batch in 0..tensor.shape().as_slice()[0] {
            debug!(?batch, "Convoluting batch");
            for out_channel in 0..weights.shape().as_slice()[0] {
                debug!(?out_channel, "Convoluting out channel");
                for out_y in 0..out_height {
                    for out_x in 0..out_width {
                        let mut sum = 0.0f32;

                        for in_channel in 0..tensor.shape().as_slice()[1] {
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let in_y = out_y * stride.as_slice()[0] + ky;
                                    let in_x = out_x * stride.as_slice()[1] + kx;

                                    // Apply padding offset with bounds checking
                                    let in_y_with_pad = in_y as isize - pad_h as isize;
                                    let in_x_with_pad = in_x as isize - pad_w as isize;

                                    // Check bounds (padding adds zeros outside bounds)
                                    if in_y_with_pad >= 0
                                        && in_x_with_pad >= 0
                                        && (in_y_with_pad as usize) < in_height
                                        && (in_x_with_pad as usize) < in_width
                                    {
                                        let input_index = batch
                                            * tensor.shape().size_for_dimension(1)
                                            + in_channel * tensor.shape().size_for_dimension(2)
                                            + (in_y_with_pad as usize)
                                                * tensor.shape().size_for_dimension(3)
                                            + (in_x_with_pad as usize);

                                        let weight_index = out_channel
                                            * weights.shape().size_for_dimension(1)
                                            + in_channel * weights.shape().size_for_dimension(2)
                                            + ky * weights.shape().size_for_dimension(3)
                                            + kx;

                                        sum += input_data[input_index] * weight_data[weight_index];
                                    }
                                    // If out of bounds due to padding, we implicitly add 0 (no operation needed)
                                }
                            }
                        }

                        let output_index = batch * new_shape.size_for_dimension(1)
                            + out_channel * new_shape.size_for_dimension(2)
                            + out_y * new_shape.size_for_dimension(3)
                            + out_x;

                        result[output_index] = sum;

                        debug!(?output_index, ?sum, "Setting result tensor value");
                    }
                }
            }
        }

        Cpu::new_tensor(new_shape, result)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use tracing_test::traced_test;

    use crate::Cpu;

    #[test]
    #[traced_test]
    fn test_conv() {
        let tensor: Tensor<Cpu, 4, f32> = Tensor::from([
            [
                [
                    [100.0, 101.0, 102.0],
                    [103.0, 104.0, 105.0],
                    [106.0, 107.0, 108.0],
                ],
                [
                    [110.0, 111.0, 112.0],
                    [113.0, 114.0, 115.0],
                    [116.0, 117.0, 118.0],
                ],
            ],
            [
                [
                    [200.0, 201.0, 202.0],
                    [203.0, 204.0, 105.0],
                    [206.0, 207.0, 208.0],
                ],
                [
                    [210.0, 211.0, 212.0],
                    [213.0, 214.0, 215.0],
                    [216.0, 217.0, 218.0],
                ],
            ],
        ]);

        let weights: Tensor<Cpu, 4, f32> = Tensor::from([
            [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]],
            [[[3.0, 3.0], [3.0, 3.0]], [[4.0, 4.0], [4.0, 4.0]]],
        ]);

        let result = tensor
            .conv_2d(
                &weights,
                Conv2DParams {
                    stride: Shape([1, 1]),
                    padding: Shape([0, 0]),
                },
            )
            .unwrap();

        let shape = result.shape().to_owned();

        assert_eq!(shape, [2, 2, 2, 2].into());

        assert_eq!(
            result.to_vec(),
            &[
                1304.0, 1316.0, 1340.0, 1352.0, 3016.0, 3044.0, 3100.0, 3128.0, 2504.0, 2416.0,
                2540.0, 2452.0, 5816.0, 5544.0, 5900.0, 5628.0
            ]
        );
    }

    #[test]
    #[traced_test]
    fn test_sobel() {
        let tensor: Tensor<Cpu, 4, f32> =
            Tensor::from([[[[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]]]);

        let weights: Tensor<Cpu, 4, f32> =
            Tensor::from([[[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]]);

        let result = tensor.conv_2d(&weights, Conv2DParams::default()).unwrap();

        assert_eq!(result.shape(), &Shape([1, 1, 1, 1]));
        println!("CPU Sobel result: {:?}", result.to_vec());
        assert_eq!(&result.to_vec(), &[-3.0]);
    }
}
