use std::{collections::HashMap, sync::Mutex};

use lazy_static::lazy_static;
use tracing::debug;

use afterburner_core::prelude::*;

lazy_static! {
    static ref DATA: Mutex<HashMap<usize, Vec<u8>>> = Mutex::new(HashMap::new());
}

#[derive(Clone, Debug)]
pub struct Cpu;

impl Cpu {
    pub(crate) fn tensor_from_raw<const D: usize, T: Clone>(
        data: &'static [T],
        shape: Shape<D>,
    ) -> Tensor<Self, D, T> {
        let data = unsafe { Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.len()) };
        Cpu::new_tensor(shape, data)
    }

    fn tensor<const D: usize, const L: usize, T: Clone>(
        &self,
        shape: Shape<D>,
        data: [T; L],
    ) -> Tensor<Self, D, T> {
        todo!()
    }
}

impl Backend for Cpu {
    fn as_slice<const D: usize, T: Clone>(t: &Tensor<Self, D, T>) -> &[T] {
        let lock = DATA.lock().unwrap();
        let data = lock.get(&t.id).unwrap().as_slice();
        let ptr = data.as_ptr() as *mut T;
        let length = data.len() / std::mem::size_of::<T>();

        unsafe { std::slice::from_raw_parts(ptr, length) }
    }

    fn new_tensor<const D: usize, T: Clone>(shape: Shape<D>, vec: Vec<T>) -> Tensor<Self, D, T> {
        let t = Tensor::new(Self, shape);

        let length = std::mem::size_of::<T>() * vec.len();
        let data = unsafe { Vec::from_raw_parts(vec.as_ptr() as *mut u8, length, length) };

        // forget vec since our new vec has ownership of memory now thorugh the pointer magic above
        std::mem::forget(vec);

        let mut lock = DATA.lock().unwrap();
        lock.insert(t.id, data);

        t
    }
}

impl Conv2DImpl<Cpu, f32> for Cpu {
    fn conv_2d(
        &self,
        tensor: &Tensor<Cpu, 4, f32>,
        weights: &Tensor<Cpu, 4, f32>,
        stride: Shape<2>,
    ) -> Tensor<Cpu, 4, f32> {
        debug!(?tensor, ?weights, ?stride, "Convoluting tensor");

        let new_shape: Shape<4> = [
            tensor.shape().as_slice()[0],
            weights.shape().as_slice()[1],
            tensor.shape().as_slice()[2] / stride.as_slice()[0] - weights.shape().as_slice()[2] / 2,
            tensor.shape().as_slice()[3] / stride.as_slice()[1] - weights.shape().as_slice()[3] / 2,
        ]
        .into();

        // new tensor
        let mut result: Vec<f32> = vec![0f32; new_shape.size()];

        for out_channel in 0..weights.shape().as_slice()[0] {
            debug!(?out_channel, "Convoluting out channel");
            for batch in 0..tensor.shape().as_slice()[0] {
                debug!(?batch, "Convoluting batch");
                for channel in 0..tensor.shape().as_slice()[1] {
                    debug!(?channel, "Convoluting channel");
                    for tensor_y in
                        0..tensor.shape().as_slice()[2] - weights.shape().as_slice()[2] / 2
                    {
                        for tensor_x in
                            0..tensor.shape().as_slice()[3] - weights.shape().as_slice()[3] / 2
                        {
                            for weight_y in 0..weights.shape().as_slice()[2] {
                                for weight_x in 0..weights.shape().as_slice()[3] {
                                    let output_index = batch * new_shape.size_for_dimension(1)
                                        + out_channel * new_shape.size_for_dimension(2)
                                        + tensor_y * new_shape.size_for_dimension(3)
                                        + tensor_x;

                                    let tensor_index = batch * tensor.shape().size_for_dimension(1)
                                        + channel * tensor.shape().size_for_dimension(2)
                                        + tensor_y * tensor.shape().size_for_dimension(3)
                                        + tensor_x
                                        + weight_y * tensor.shape().size_for_dimension(3)
                                        + weight_x;

                                    let weight_index = out_channel
                                        * weights.shape().size_for_dimension(1)
                                        + channel * weights.shape().size_for_dimension(2)
                                        + weight_y * weights.shape().size_for_dimension(3)
                                        + weight_x;

                                    let weight = weights.as_slice()[weight_index];

                                    let value = result[output_index]
                                        + tensor.as_slice()[tensor_index] * weight;

                                    result[output_index] = value;

                                    debug!(
                                        ?output_index,
                                        ?weight,
                                        ?weight_index,
                                        ?value,
                                        "Updating result tensor"
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        let tensor = Tensor::new(Cpu, new_shape);

        {
            let mut data = DATA.lock().unwrap();
            data.insert(tensor.id, as_bytes(result));
        }

        tensor
    }
}

fn as_bytes<T: Clone + Sized>(data: Vec<T>) -> Vec<u8> {
    unsafe {
        let length = std::mem::size_of::<T>() * data.len();
        Vec::from_raw_parts(data.as_ptr() as *mut u8, length, length)
    }
}

fn as_slice<T: Clone + Sized>(data: &[u8]) -> &[T] {
    unsafe {
        let ptr = data.as_ptr() as *mut T;
        let length = data.len() / std::mem::size_of::<T>();
        std::slice::from_raw_parts(ptr, 5)
    }
}

#[cfg(test)]
mod test {
    use afterburner_core::prelude::*;
    use tracing_test::traced_test;

    use crate::Cpu;

    #[test]
    fn test_from_4d() {
        let tensor: Tensor<Cpu, 4, _> = Tensor::from([
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

        assert_eq!(
            tensor.as_slice(),
            &[
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 110.0, 111.0, 112.0,
                113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 200.0, 201.0, 202.0, 203.0, 204.0, 105.0,
                206.0, 207.0, 208.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0,
            ]
        );
    }

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
