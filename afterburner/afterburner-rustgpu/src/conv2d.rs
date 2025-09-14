use afterburner_core::prelude::*;
use afterburner_ops::prelude::*;
use afterburner_rustgpu_shared::{conv2d::RustGpuConv2DParams, RustGpuShape};
use tracing::debug;

use crate::{run_with_backend, RustGpu};

impl Conv2DBackend<RustGpu, f32> for RustGpu {
    fn conv_2d(
        tensor: &Tensor<RustGpu, 4, f32>,
        weights: &Tensor<RustGpu, 4, f32>,
        params: Conv2DParams,
    ) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            debug!(?tensor, ?weights, ?params, "conv_2d");

            let stride = params.stride;
            let padding = params.padding;

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

            let output = Tensor::create(new_shape);

            // Initialize output buffer with zeros to ensure correct computation
            let zero_data = vec![0.0f32; new_shape.size()];
            backend.create_buffer_init(output.id, zero_data).unwrap();

            let params = RustGpuConv2DParams {
                dimensions: RustGpuShape([
                    tensor.shape.0[0] as u32,
                    tensor.shape.0[1] as u32,
                    tensor.shape.0[2] as u32,
                    tensor.shape.0[3] as u32,
                ]),
                conv: RustGpuShape([
                    weights.shape.0[0] as u32,
                    weights.shape.0[1] as u32,
                    weights.shape.0[2] as u32,
                    weights.shape.0[3] as u32,
                ]),
                stride: RustGpuShape([stride.0[0] as u32, stride.0[1] as u32]),
                padding: RustGpuShape([padding.0[0] as u32, padding.0[1] as u32]),
            };

            backend.run_shader_2("conv2d", tensor.id, weights.id, output.id, params);
            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_shader_basic_write() {
        init();

        // Test if the shader can write a simple value to the output buffer
        let tensor: Tensor<RustGpu, 4, f32> =
            Tensor::from([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]);

        let weights: Tensor<RustGpu, 4, f32> =
            Tensor::from([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]);

        println!("Test input data: {:?}", tensor.to_vec());
        println!("Test weights data: {:?}", weights.to_vec());

        let result = tensor.conv_2d(&weights, Conv2DParams::default()).unwrap();

        println!("Test result shape: {:?}", result.shape());
        println!("Test result data: {:?}", result.to_vec());

        // This should produce 45.0 (sum of all input elements: 1+2+3+4+5+6+7+8+9 = 45)
        assert_eq!(result.shape(), &Shape([1, 1, 1, 1]));
        assert_eq!(&result.to_vec(), &[45.0]);
    }

    #[test]
    fn test_sobel() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> =
            Tensor::from([[[[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]]]);

        let weights: Tensor<RustGpu, 4, f32> =
            Tensor::from([[[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]]]);

        println!("Input tensor shape: {:?}", tensor.shape());
        println!("Input tensor data: {:?}", tensor.to_vec());
        println!("Weights shape: {:?}", weights.shape());
        println!("Weights data: {:?}", weights.to_vec());

        let result = tensor.conv_2d(&weights, Conv2DParams::default()).unwrap();

        println!("Result shape: {:?}", result.shape());
        println!("Result data: {:?}", result.to_vec());

        assert_eq!(result.shape, Shape([1, 1, 1, 1]));
        assert_eq!(&result.to_vec(), &[-3.0]);
    }

    #[test]
    #[tracing_test::traced_test]
    fn test_convo() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([
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

        let weights: Tensor<RustGpu, 4, f32> = Tensor::from([
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
}
