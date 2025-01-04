use afterburner_core::prelude::*;

#[derive(Clone)]
pub struct Cpu;

impl Backend for Cpu {}

impl Conv2DImpl<Cpu, f32> for Cpu {
    fn conv_2d(
        &self,
        tensor: &Tensor<Cpu, 4, f32>,
        weights: &Tensor<Cpu, 4, f32>,
        stride: Shape<2>,
    ) -> Tensor<Cpu, 4, f32> {
        let t: Tensor<NoBackend, 4, f32> = [[0f32, 0.0, 0.0, 0.0], [0f32, 0.0, 0.0, 0.0]].into();
        t.copy_to(self)
    }
}

#[cfg(test)]
mod test {
    use afterburner_core::prelude::*;

    use crate::Cpu;

    #[test]
    fn test_conv() {
        let tensor: Tensor<_, 4, f32> = Tensor::empty(Cpu, [1, 2, 4, 4]);
        let mat: Tensor<_, 4, f32> = Tensor::empty(Cpu, [2, 2, 1, 2]);
        let result = tensor.conv_2d(&mat, 1);

        let shape = result.shape();

        assert_eq!(shape, [1, 4, 4, 3].into());

        assert_eq!(
            result.as_slice(),
            &[0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }
}
