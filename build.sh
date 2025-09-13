#!/bin/bash

# Build script for afterburner-ai that handles C++ compilation issues
# Sets CXXFLAGS to avoid treating warnings as errors

set -e

echo "Building afterburner-ai with CXXFLAGS override..."

# Change to the afterburner directory
cd afterburner

# Set environment variables to avoid C++ compilation errors
export CXXFLAGS="-Wno-error"
export CFLAGS="-Wno-error"

# Run cargo build
cargo build "$@"

echo "Build completed successfully!"
