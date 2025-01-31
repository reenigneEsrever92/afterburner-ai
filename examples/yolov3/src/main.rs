use afterburner_rustgpu::prelude::*;

fn main() {
    let t: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0f32]]]]);
    println!("{t:?}");
}
