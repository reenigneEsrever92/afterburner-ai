use std::sync::Mutex;

use afterburner_core::prelude::*;

const SHADER: &[u8] = include_bytes!(env!("shaders.spv"));

#[derive(Debug, thiserror::Error)]
enum RGError {}

#[derive(Debug)]
struct RustGpuBackend {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
}

static BACKEND: Mutex<Option<RustGpuBackend>> = Mutex::new(None);

#[derive(Debug, Clone)]
struct RustGpu {}

impl Backend for RustGpu {
    fn as_slice<const D: usize, T: Clone>(t: &Tensor<Self, D, T>) -> &[T] {
        let lock = BACKEND.lock().unwrap();

        if let Some(backend) = lock.as_ref() {
            todo!()
        } else {
            panic!("RustGpu backend has not been initialized call init() first!");
        }
    }

    fn new_tensor<const D: usize, T: Clone>(shape: Shape<D>, data: Vec<T>) -> Tensor<Self, D, T> {
        todo!()
    }
}

pub fn init() {
    let backend = smol::block_on(async move {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::SpirV(wgpu::util::make_spirv_raw(SHADER)),
        });

        RustGpuBackend {
            adapter,
            device,
            queue,
            shader,
        }
    });

    let mut lock = BACKEND.lock().unwrap();
    *lock = Some(backend);
}

impl Conv2DImpl<RustGpu, f32> for RustGpu {
    fn conv_2d(
        &self,
        tensor: &Tensor<RustGpu, 4, f32>,
        weights: &Tensor<RustGpu, 4, f32>,
        params: Conv2DParams,
    ) -> Tensor<RustGpu, 4, f32> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use afterburner_core::prelude::*;

    use crate::RustGpu;

    // #[test]
    // fn test_conv() {
    //     let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([
    //         [
    //             [
    //                 [100.0, 101.0, 102.0],
    //                 [103.0, 104.0, 105.0],
    //                 [106.0, 107.0, 108.0],
    //             ],
    //             [
    //                 [110.0, 111.0, 112.0],
    //                 [113.0, 114.0, 115.0],
    //                 [116.0, 117.0, 118.0],
    //             ],
    //         ],
    //         [
    //             [
    //                 [200.0, 201.0, 202.0],
    //                 [203.0, 204.0, 105.0],
    //                 [206.0, 207.0, 208.0],
    //             ],
    //             [
    //                 [210.0, 211.0, 212.0],
    //                 [213.0, 214.0, 215.0],
    //                 [216.0, 217.0, 218.0],
    //             ],
    //         ],
    //     ]);

    //     let weights: Tensor<RustGpu, 4, f32> = Tensor::from([
    //         [[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]],
    //         [[[3.0, 3.0], [3.0, 3.0]], [[4.0, 4.0], [4.0, 4.0]]],
    //     ]);

    //     let result = tensor.conv_2d(&weights, Conv2DParams::default()).unwrap();

    //     let shape = result.shape().to_owned();

    //     assert_eq!(shape, [2, 2, 2, 2].into());

    //     assert_eq!(
    //         result.as_slice(),
    //         &[
    //             1304.0, 1316.0, 1340.0, 1352.0, 3016.0, 3044.0, 3100.0, 3128.0, 2504.0, 2416.0,
    //             2540.0, 2452.0, 5816.0, 5544.0, 5900.0, 5628.0
    //         ]
    //     );
    // }
}
