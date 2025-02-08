use std::{collections::HashMap, ops::Deref, sync::Mutex};

use afterburner_core::prelude::*;
use afterburner_rustgpu_shared::{RustGpuConv2DParams, RustGpuShape};
use bytemuck::{bytes_of, Pod};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupEntry, Buffer, BufferDescriptor, BufferUsages, Features, Limits,
};

pub mod nn;
pub mod prelude;

const SHADER: &[u8] = include_bytes!(env!("shaders.spv"));

static BACKEND: Mutex<Option<RustGpuBackend>> = Mutex::new(None);

#[derive(Debug)]
pub struct RustGpuBackend {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    buffers: HashMap<usize, Buffer>,
}

impl RustGpuBackend {
    pub fn create_buffer(&mut self, id: usize, size: usize) -> AbResult<()> {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(&format!("Compute Buffer [ id = {id} ]")),
            usage: BufferUsages::COPY_SRC | BufferUsages::UNIFORM | BufferUsages::STORAGE,
            size: size as u64,
            mapped_at_creation: false,
        });

        self.buffers.insert(id, buffer);

        Ok(())
    }

    pub fn create_buffer_init<T: Clone>(&mut self, id: usize, data: Vec<T>) -> AbResult<()> {
        let contents = cast_slice(data.as_slice());

        let buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some(&format!("Compute Buffer [ id = {id} ]")),
            usage: BufferUsages::COPY_SRC | BufferUsages::UNIFORM | BufferUsages::STORAGE,
            contents,
        });

        self.buffers.insert(id, buffer);

        Ok(())
    }

    pub fn read_buffer<const D: usize, T: Clone>(
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

        self.queue.submit([encoder.finish()]);

        let buffer_slice = output_buffer.slice(..);

        buffer_slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        let buffer_view = buffer_slice.get_mapped_range();
        let data = buffer_view.deref();
        let result = cast_slice(data).to_vec();

        Ok(result)
    }

    pub fn run_shader<T>(
        &mut self,
        entry_point: &str,
        buffer1_id: usize,
        buffer2_id: usize,
        output_buffer_id: usize,
        params: T,
    ) {
        let buffer_1 = self.buffers.get(&buffer1_id).unwrap();
        let buffer_2 = self.buffers.get(&buffer2_id).unwrap();
        let output_buffer = self.buffers.get(&output_buffer_id).unwrap();

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            count: None,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                min_binding_size: None,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            count: None,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                min_binding_size: None,
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            count: None,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                has_dynamic_offset: false,
                                min_binding_size: None,
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                            },
                        },
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<T>() as u32,
                }],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &self.shader,
                    entry_point,
                });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());

            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_push_constants(0, cast_struct(&params));
            cpass.dispatch_workgroups(64, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

#[inline]
fn cast_slice<T, R: Sized>(data: &[T]) -> &[R] {
    let ptr = data.as_ptr() as *mut R;
    let size = std::mem::size_of_val(data);
    unsafe { std::slice::from_raw_parts_mut(ptr, size / std::mem::size_of::<R>()) }
}

fn cast_struct<T>(data: &T) -> &[u8] {
    let ptr = (data as *const T) as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, std::mem::size_of::<T>()) }
}

pub struct RustGpuBuilder;

#[derive(Debug, Clone)]
pub struct RustGpu {}

impl Backend for RustGpu {
    fn read_tensor<const D: usize, T: Clone>(t: &Tensor<Self, D, T>) -> Vec<T> {
        let data = {
            let mut lock = BACKEND.lock().unwrap();

            if let Some(backend) = lock.as_mut() {
                backend.read_buffer(t).unwrap()
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

            backend.create_buffer_init(tensor.id, data).unwrap();

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
                    required_features: Features::PUSH_CONSTANTS,
                    required_limits: Limits {
                        max_push_constant_size: 128,
                        ..Default::default()
                    },
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

pub fn run_with_backend<T>(func: impl FnOnce(&mut RustGpuBackend) -> T) -> T {
    let mut lock = BACKEND.lock().unwrap();

    if let Some(backend) = lock.as_mut() {
        func(backend)
    } else {
        panic!("RustGpu backend has not been initialized call init() first!");
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
