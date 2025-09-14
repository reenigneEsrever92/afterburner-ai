use afterburner_core::prelude::*;

use crate::{run_with_backend, RustGpu};

impl<const D: usize> ConvertImpl<RustGpu, D, u8, f32> for RustGpu {
    fn convert(input: &Tensor<RustGpu, D, u8>) -> Tensor<RustGpu, D, f32> {
        run_with_backend(|backend| {
            let t = Tensor::create(*input.shape());

            backend
                .create_buffer(t.id, t.size())
                .expect("New buffer created");

            backend.run_shader("convert_u8_f32", input.id, t.id);

            t
        })
    }
}

impl<const D: usize> ConvertImpl<RustGpu, D, f32, u8> for RustGpu {
    fn convert(input: &Tensor<RustGpu, D, f32>) -> Tensor<RustGpu, D, u8> {
        run_with_backend(|backend| {
            let t = Tensor::create(*input.shape());

            backend
                .create_buffer(t.id, t.size())
                .expect("New buffer created");

            backend.run_shader("convert_f32_u8", input.id, t.id);

            t
        })
    }
}

// Specialized image conversion functions
impl RustGpu {
    /// Convert RGB image (u8) to grayscale (f32) - specialized for 1D tensors
    pub fn rgb_to_grayscale(
        input: &Tensor<RustGpu, 1, u8>,
        width: usize,
        height: usize,
    ) -> Tensor<RustGpu, 1, f32> {
        run_with_backend(|backend| {
            let total_pixels = width * height;
            let grayscale_shape = Shape([total_pixels]);

            let t = Tensor::create(grayscale_shape);

            backend
                .create_buffer(t.id, t.size())
                .expect("New buffer created");

            backend.run_shader("convert_rgb_to_grayscale", input.id, t.id);

            t
        })
    }

    /// Convert grayscale (f32) to RGB image (u8) - specialized for 1D tensors
    pub fn grayscale_to_rgb(
        input: &Tensor<RustGpu, 1, f32>,
        width: usize,
        height: usize,
    ) -> Tensor<RustGpu, 1, u8> {
        run_with_backend(|backend| {
            let total_pixels = width * height;
            let rgb_shape = Shape([total_pixels * 3]);

            let t = Tensor::create(rgb_shape);

            backend
                .create_buffer(t.id, t.size())
                .expect("New buffer created");

            backend.run_shader("convert_grayscale_to_rgb", input.id, t.id);

            t
        })
    }
}

#[cfg(test)]
mod test {
    use afterburner_core::{backend::Backend, prelude::Convert, prelude::Shape, tensor::Tensor};
    use tracing_test::traced_test;

    use crate::{init, RustGpu};

    #[test]
    #[traced_test]
    fn test_convert_u8_f32() {
        init();

        let t: Tensor<RustGpu, 1, u8> = [1, 2, 3].into();

        let converted: Tensor<_, 1, f32> = t.convert();

        assert_eq!(converted.to_vec(), [1.0f32, 2.0, 3.0]);
    }

    #[test]
    #[traced_test]
    fn test_convert_f32_u8() {
        init();

        let t: Tensor<RustGpu, 1, f32> = RustGpu::new_tensor(Shape([3]), vec![1.5, 255.7, 0.2]);

        let converted: Tensor<_, 1, u8> = t.convert();

        assert_eq!(converted.to_vec(), [1u8, 255, 0]);
    }
}
