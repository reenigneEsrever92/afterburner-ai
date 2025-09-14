// use clap::Parser;
use spirv_builder::SpirvBuilder;

// #[derive(Parser, Debug)]
// #[command(version, about, long_about = None)]
// struct Args {
//     #[arg(required = true)]
//     path: String,
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // println!("In process vars: {:?}", std::env::vars());
    // let args = Args::parse();
    // std::env::set_current_dir(args.path);
    let result = SpirvBuilder::new("afterburner-shaders", "spirv-unknown-vulkan1.4")
        .capability(spirv_builder::Capability::Int8)
        .build()?;

    // Set the environment variable for the build script to find the compiled shader
    println!(
        "cargo:rustc-env=afterburner_shaders.spv={}",
        result.module.unwrap_single().display()
    );

    Ok(())
}
