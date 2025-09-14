use std::{collections::HashMap, ops::Deref, sync::Mutex};

use lazy_static::lazy_static;

use afterburner_core::prelude::*;

pub mod batch_norm;
pub mod conv2d;
pub mod prelude;

lazy_static! {
    static ref DATA: Mutex<HashMap<usize, Vec<u8>>> = Mutex::new(HashMap::new());
}

#[derive(Clone, Debug)]
pub struct Cpu;

impl Backend for Cpu {
    fn read_tensor<const D: usize, T: Clone>(t: &Tensor<Self, D, T>) -> Vec<T> {
        let lock = DATA.lock().unwrap();
        let data = lock.get(&t.id).unwrap().as_slice();
        let ptr = data.as_ptr() as *mut T;
        let length = data.len() / std::mem::size_of::<T>();

        unsafe { std::slice::from_raw_parts(ptr, length).to_vec() }
    }

    fn new_tensor<const D: usize, T: Clone>(shape: Shape<D>, vec: Vec<T>) -> Tensor<Self, D, T> {
        let t = Tensor::create(shape);

        let length = std::mem::size_of::<T>() * vec.len();
        let data = unsafe { Vec::from_raw_parts(vec.as_ptr() as *mut u8, length, length) };

        // forget vec since our new vec has ownership of memory now thorugh the pointer magic above
        std::mem::forget(vec);

        let mut lock = DATA.lock().unwrap();
        lock.insert(t.id, data);

        t
    }

    fn delete_tensor<const D: usize, T: Clone>(t: &mut Tensor<Self, D, T>) {
        let mut lock = DATA.lock().unwrap();
        lock.remove(&t.id);
    }
}

#[cfg(test)]
mod test {
    use afterburner_core::prelude::*;
    use afterburner_ops::prelude::*;
    use tracing_test::traced_test;

    use crate::Cpu;

    #[test]
    #[traced_test]
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
            tensor.to_vec(),
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

        let result = tensor.conv_2d(&weights, Conv2DParams::default()).unwrap();

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
