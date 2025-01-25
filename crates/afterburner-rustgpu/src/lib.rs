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
