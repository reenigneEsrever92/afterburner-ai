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

            backend.run_shader_2(
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

// #[cfg(test)]
// mod test {
//     use crate::prelude::*;

//     #[test]
//     fn test_conv() {
//         init();
//         let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([
//             [
//                 [
//                     [100.0, 101.0, 102.0],
//                     [103.0, 104.0, 105.0],
//                     [106.0, 107.0, 108.0],
//                 ],
//                 [
//                     [110.0, 111.0, 112.0],
//                     [113.0, 114.0, 115.0],
//                     [116.0, 117.0, 118.0],
//                 ],
//             ],
//             [
//                 [
//                     [200.0, 201.0, 202.0],
//                     [203.0, 204.0, 105.0],
//                     [206.0, 207.0, 208.0],
//                 ],
//                 [
//                     [210.0, 211.0, 212.0],
//                     [213.0, 214.0, 215.0],
//                     [216.0, 217.0, 218.0],
//                 ],
//             ],
//         ]);

//         let weights: Tensor<RustGpu, 4, f32> = Tensor::from([
//             [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]],
//             [[[3.0, 3.0], [3.0, 3.0]], [[4.0, 4.0], [4.0, 4.0]]],
//         ]);

//         let result = tensor.conv_2d(&weights, Conv2DParams::default()).unwrap();

//         let shape = result.shape().to_owned();

//         assert_eq!(shape, [2, 2, 2, 2].into());

//         assert_eq!(
//             result.to_vec(),
//             &[
//                 1304.0, 1316.0, 1340.0, 1352.0, 3016.0, 3044.0, 3100.0, 3128.0, 2504.0, 2416.0,
//                 2540.0, 2452.0, 5816.0, 5544.0, 5900.0, 5628.0
//             ]
//         );
//     }
// }
