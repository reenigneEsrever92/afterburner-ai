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

#[cfg(test)]
mod test {
    use afterburner_core::{prelude::Convert, tensor::Tensor};
    use tracing_test::traced_test;

    use crate::{init, RustGpu};

    #[test]
    #[traced_test]
    fn test_convert() {
        init();

        let t: Tensor<RustGpu, 1, u8> = [1, 2, 3].into();

        let converted: Tensor<_, 1, f32> = t.convert();

        assert_eq!(converted.to_vec(), [1.0f32, 2.0, 3.0]);
    }
}
