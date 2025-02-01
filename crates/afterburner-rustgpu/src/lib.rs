use std::{collections::HashMap, sync::Mutex};

use afterburner_core::prelude::*;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferDescriptor, BufferUsages,
};

pub mod prelude;

const SHADER: &[u8] = include_bytes!(env!("shaders.spv"));

static BACKEND: Mutex<Option<RustGpuBackend>> = Mutex::new(None);

#[derive(Debug)]
struct RustGpuBackend {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    buffers: HashMap<usize, Buffer>,
}

impl RustGpuBackend {
    fn create_buffer<T: Clone>(&mut self, id: usize, mut data: Vec<T>) -> AbResult<()> {
        let buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("Compute Buffer [ id = {id} ]")),
            usage: BufferUsages::COPY_SRC | BufferUsages::UNIFORM,
            contents: cast_slice(data.as_mut_slice()),
        });

        self.buffers.insert(id, buffer);

        Ok(())
    }

    fn read_buffer<const D: usize, T: Clone>(
        &mut self,
        t: &Tensor<RustGpu, D, T>,
    ) -> AbResult<Vec<T>> {
        let output_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: t.size() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let tensor_buffer = self.buffers.get(&t.id).unwrap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(tensor_buffer, 0, &output_buffer, 0, t.size() as u64);

        encoder.finish();

        let buffer_slice = output_buffer.slice(..);

        buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        let buffer_view = buffer_slice.get_mapped_range();
        let mut data = buffer_view.to_vec();
        let result = cast_slice(data.as_mut_slice()).to_vec();

        Ok(result)
    }
}

#[inline]
fn cast_slice<T, R: Sized>(data: &mut [T]) -> &mut [R] {
    let ptr = data.as_mut_ptr() as *mut R;
    let size = std::mem::size_of_val(data);
    unsafe { std::slice::from_raw_parts_mut(ptr, size / std::mem::size_of::<R>()) }
}

pub struct RustGpuBuilder;

#[derive(Debug, Clone)]
pub struct RustGpu {}

impl Backend for RustGpu {
    fn read_tensor<const D: usize, T: Clone>(t: &Tensor<Self, D, T>) -> Vec<T> {
        let data = {
            let mut lock = BACKEND.lock().unwrap();

            if let Some(backend) = lock.as_mut() {
                backend.read_buffer(&t).unwrap()
            } else {
                panic!("RustGpu backend has not been initialized call init() first!");
            }
        };

        data
    }

    fn new_tensor<const D: usize, T: Clone>(shape: Shape<D>, data: Vec<T>) -> Tensor<Self, D, T> {
        let mut lock = BACKEND.lock().unwrap();

        if let Some(backend) = lock.as_mut() {
            let tensor = Tensor::create(shape);

            backend.create_buffer(tensor.id, data).unwrap();

            tensor
        } else {
            panic!("RustGpu backend has not been initialized call init() first!");
        }
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
            buffers: HashMap::new(),
        }
    });

    let mut lock = BACKEND.lock().unwrap();
    *lock = Some(backend);
}

impl Conv2DImpl<RustGpu, f32> for RustGpu {
    fn conv_2d(
        tensor: &Tensor<RustGpu, 4, f32>,
        weights: &Tensor<RustGpu, 4, f32>,
        params: Conv2DParams,
    ) -> Tensor<RustGpu, 4, f32> {
        let mut lock = BACKEND.lock().unwrap();

        if let Some(backend) = lock.as_mut() {
            todo!()
        } else {
            panic!("RustGpu backend has not been initialized call init() first!");
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_create_tensor() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0f32]]]]);
        assert_eq!(&tensor.to_vec(), &[1.0f32]);
    }

    #[test]
    fn test_conv() {
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
