use afterburner_core::prelude::*;

#[derive(Clone, Debug)]
pub struct Cpu;

impl Backend for Cpu {}

impl Conv2DImpl<Cpu, f32> for Cpu {
    fn conv_2d(
        &self,
        tensor: &Tensor<Cpu, 4, f32>,
        weights: &Tensor<Cpu, 4, f32>,
        stride: Shape<2>,
    ) -> Tensor<Cpu, 4, f32> {
        tracing::debug!(?tensor, ?weights, "Convoluting tensor");
        for b in 0..tensor.shape().as_slice()[0] {
            tracing::debug!(?b, "Convoluting batch");
        }
        todo!()
    }
}

#[cfg(test)]
mod test {
    use afterburner_core::prelude::*;
    use tracing_test::traced_test;

    use crate::Cpu;

    #[test]
    #[traced_test]
    fn test_conv() {
        let tensor = Tensor::from([
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
        ])
        .copy_to(Cpu);

        let weights = Tensor::from([
            [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]],
            [[[3.0, 3.0], [3.0, 3.0]], [[4.0, 4.0], [4.0, 4.0]]],
        ])
        .copy_to(Cpu);

        let result = tensor.conv_2d(&weights, 1).unwrap();

        let shape = result.shape().to_owned();

        assert_eq!(shape, [2, 2, 2, 2].into());

        assert_eq!(
            result.as_slice(),
            &[
                1304.0, 1316.0, 1340.0, 1352.0, 3016.0, 3044.0, 3100.0, 3128.0, 2504.0, 2416.0,
                2540.0, 2452.0, 5816.0, 5544.0, 5900.0, 5628.0
            ]
        );
    }
}
