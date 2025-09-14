use std::{collections::HashMap, ops::Deref, sync::Mutex};

use afterburner_core::prelude::*;
use tracing::debug;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferDescriptor, BufferUsages, Features, Limits, PipelineCompilationOptions,
};

pub mod batch_norm;
pub mod channel_normalize;
pub mod conv2d;
pub mod convert;
pub mod normalize;
pub mod prelude;

const SHADER: &[u8] = include_bytes!(env!("afterburner_shaders.spv"));

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

    pub fn delete_buffer(&mut self, id: usize) {
        self.buffers.remove(&id);
    }

    pub fn run_shader(&mut self, entry_point: &str, buffer1_id: usize, output_buffer_id: usize) {
        let buffer_1 = self.buffers.get(&buffer1_id).unwrap();
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
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Compute Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &self.shader,
                    entry_point: Some(entry_point),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());

            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&compute_pipeline);
            self.safe_dispatch_workgroups(&mut cpass, output_buffer_id, 4);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_shader_with_params<T>(
        &mut self,
        entry_point: &str,
        buffer1_id: usize,
        output_buffer_id: usize,
        params: T,
    ) {
        let buffer_1 = self.buffers.get(&buffer1_id).unwrap();
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
                    entry_point: Some(entry_point),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());

            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_push_constants(0, cast_struct(&params));
            self.safe_dispatch_workgroups(&mut cpass, output_buffer_id, 4);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_shader_2<T>(
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
                    entry_point: Some(entry_point),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());

            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_push_constants(0, cast_struct(&params));
            self.safe_dispatch_workgroups(&mut cpass, output_buffer_id, 4);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_shader_3<T>(
        &mut self,
        entry_point: &str,
        buffer1_id: usize,
        buffer2_id: usize,
        buffer3_id: usize,
        output_buffer_id: usize,
        params: T,
    ) {
        let buffer_1 = self.buffers.get(&buffer1_id).unwrap();
        let buffer_2 = self.buffers.get(&buffer2_id).unwrap();
        let buffer_3 = self.buffers.get(&buffer3_id).unwrap();
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                            },
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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
                    resource: buffer_3.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
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
                    entry_point: Some(entry_point),
                    compilation_options: PipelineCompilationOptions::default(),
                    cache: None,
                });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());

            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_push_constants(0, cast_struct(&params));

            self.safe_dispatch_workgroups(&mut cpass, output_buffer_id, 4);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

impl RustGpuBackend {
    fn safe_dispatch_workgroups(
        &self,
        cpass: &mut wgpu::ComputePass,
        output_buffer_id: usize,
        element_size: usize,
    ) {
        let data_size = self
            .buffers
            .get(&output_buffer_id)
            .map(|b| b.size())
            .unwrap_or(0) as usize;
        let elements = if data_size > 0 {
            data_size / element_size
        } else {
            1
        };
        let workgroups = (elements + 63) / 64; // 64 threads per workgroup, round up

        debug!(
            "Dispatch info: buffer_id={}, data_size={}, element_size={}, elements={}, workgroups={}",
            output_buffer_id, data_size, element_size, elements, workgroups
        );

        // GPU limit is 65535 workgroups per dimension, use 2D dispatch if needed
        let max_workgroups = 65535u32;
        let (x_workgroups, y_workgroups) = if workgroups <= max_workgroups as usize {
            (workgroups as u32, 1)
        } else {
            let x = max_workgroups;
            let y = ((workgroups + max_workgroups as usize - 1) / max_workgroups as usize) as u32;
            (x, y)
        };

        debug!(
            "Dispatching workgroups: x={}, y={}, z=1",
            x_workgroups, y_workgroups
        );

        cpass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
    }
}

#[inline]
fn cast_slice<T, R: Sized>(data: &[T]) -> &[R] {
    let ptr = data.as_ptr() as *mut R;
    let size = std::mem::size_of_val(data);
    unsafe { std::slice::from_raw_parts_mut(ptr, size / std::mem::size_of::<R>()) }
}

#[inline]
fn cast_struct<T>(data: &T) -> &[u8] {
    let ptr = data as *const T as *const u8;
    let size = std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, size) }
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

    fn delete_tensor<const D: usize, T: Clone>(t: &mut Tensor<Self, D, T>) {
        let mut lock = BACKEND.lock().unwrap();

        if let Some(backend) = lock.as_mut() {
            backend.delete_buffer(t.id);
        } else {
            panic!("RustGpu backend has not been initialized call init() first!");
        }
    }

    fn move_tensor<const D: usize, T: Clone, const D2: usize>(
        t: Tensor<Self, D, T>,
        shape: Shape<D2>,
    ) -> Tensor<Self, D2, T> {
        let mut lock = BACKEND.lock().unwrap();

        if let Some(_backend) = lock.as_mut() {
            let mut tensor = Tensor::create(shape);

            tensor.id = t.id;

            // forget the old tensor, otherwise it will be deleted when the old tensor is dropped
            std::mem::forget(t);

            tensor
        } else {
            panic!("RustGpu backend has not been initialized call init() first!");
        }
    }
}

pub fn init() {
    let backend = smol::block_on(async move {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
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
                    required_features: Features::PUSH_CONSTANTS
                        | Features::SPIRV_SHADER_PASSTHROUGH,
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

        let shader = unsafe {
            device.create_shader_module_spirv(&wgpu::ShaderModuleDescriptorSpirV {
                label: Some("Shader"),
                source: wgpu::util::make_spirv_raw(SHADER),
            })
        };

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
}
