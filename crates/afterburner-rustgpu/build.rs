use std::{
    os::unix::process::CommandExt,
    process::{Command, Stdio},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = std::env::current_dir()?;
    let mut build_dir = current_dir.clone();
    build_dir.push("../shaders");
    let build_dir = build_dir.canonicalize()?;

    println!("Current dir: {:?}, build dir: {:?}", current_dir, build_dir);
    println!("Env: {:?}", std::env::vars());

    let mut cargo = std::process::Command::new("cargo");

    for (key, value) in std::env::vars() {
        if key.contains("CARGO") || key.contains("RUST") {
            cargo.env_remove(key);
        }
    }

    let status = cargo
        .current_dir(build_dir)
        .args(["run", "-r", "-p", "afterburner-ai-builder"])
        .status()?;

    if !status.success() {
        if let Some(code) = status.code() {
            std::process::exit(code);
        } else {
            std::process::exit(1);
        }
    }

    Ok(())
}
