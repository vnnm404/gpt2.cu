# Build Instructions

## Build Modes

This project supports two build modes:

### Debug Mode
Debug mode includes:
- Debug symbols (`-g`)
- No optimization (`-O0`)
- AddressSanitizer (ASan) for detecting memory errors
- UndefinedBehaviorSanitizer (UBSan) for catching undefined behavior
- Extra compiler warnings
- CUDA debug symbols (`-G`)

To build in Debug mode:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

### Release Mode
Release mode includes:
- Maximum optimization (`-O3`)
- Native CPU architecture optimizations (`-march=native`)
- Fast math for CUDA (`--use_fast_math`)
- NDEBUG macro defined

To build in Release mode:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Default Build Type
If you don't specify a build type, the default is **Release**.

## Running with Sanitizers

When running programs built in Debug mode, the sanitizers are automatically enabled. If you encounter sanitizer reports, they will show detailed information about memory issues or undefined behavior.

You can control sanitizer behavior with environment variables:
```bash
# Suppress specific sanitizer checks (example)
export ASAN_OPTIONS=detect_leaks=0

# Get more verbose output
export ASAN_OPTIONS=verbosity=1

# Continue after finding errors (useful for testing)
export ASAN_OPTIONS=halt_on_error=0
```

## Clean Rebuild

To switch between build modes, it's recommended to clean the build directory:
```bash
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=<Debug|Release> ..
make
```

## Notes

- Sanitizers (ASan/UBSan) are only applied to C code, not CUDA code, as CUDA has its own debugging tools
- On Windows, sanitizers are not enabled automatically
- Debug builds will be significantly slower than Release builds due to lack of optimization and sanitizer overhead
