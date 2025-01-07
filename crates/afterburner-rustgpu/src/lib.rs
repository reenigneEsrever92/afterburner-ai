use afterburner_core::prelude::*;

const SHADER: &[u8] = include_bytes!(env!("afterburner_rustgpu_shaders.spv"));

#[derive(Debug, thiserror::Error)]
enum RGError {

}

struct RustGpu {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Backend for RustGpu {}

impl RustGpu {
    fn new() -> Self {
        smol::block_on(async move {
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

            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    ..Default::default()
                },
                None,
            );

            RustGpu {
                adapter,
                device,
                queue,
            }
        })
    }

    fn run_kernel_2d() ->
}
