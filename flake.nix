{
  inputs = {
    # Use unstable so we can access newer CUDA (12.8, 13.x) via cudaPackages_12_8 / cudaPackages_13.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            # allow unfree cuda stack from nixpkgs
            allowUnfree = true;
            # enables cuda hooks and makes cudaPackages produce cuda-enabled builds
            cudaSupport = true;

            # dont set cudaCapabilities unless to reduce build time or target specific GPU archs
            # ref: https://nixos.org/manual/nixpkgs/stable/#cuda-configuring-nixpkgs-for-cuda
            cudaCapabilities = [ "8.6" ];

            # forward compat produces PTX for future GPUs
            # maybe set to false if targeting specific arch and want minimized build outputs
            # cudaForwardCompat = true;
          };
        };

        # Explicitly select cuda toolchain and a compatible host gcc
        cudaPackages = pkgs.cudaPackages_13;
        gccHost = pkgs.gcc13;

        generateClangd = pkgs.writeShellApplication {
          name = "generate-clangd";
          runtimeInputs = with pkgs; [
            git
            stdenv.cc.cc
            findutils
            gnugrep
            gnused
            coreutils
          ];
          text = ''
            #!/usr/bin/env bash
            set -euo pipefail

            # Move to repo root
            root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
              echo "not inside a git repository" >&2
              exit 1
            }
            cd "$root"

            echo "Generating .clangd configuration in $root"

            # cpp stdlib paths (use whatever C++ compiler Nix stdenv provides)
            : "\${CXX:=c++}"
            CXX_PATHS="$($CXX -x c++ -E -v - </dev/null 2>&1 \
              | grep '^ /' | grep 'c++' | head -2)"

            if [ -z "$CXX_PATHS" ]; then
              echo "ERROR: Could not find C++ standard library paths" >&2
              exit 1
            fi

            echo "Found C++ paths:"
            echo "$CXX_PATHS"

            # cuda headers from cudatoolkit in the Nix store
            CUDA_RUNTIME="$(find ${cudaPackages.cudatoolkit} -name 'cuda_runtime.h' 2>/dev/null | head -1 || true)"
            CUDA_INCLUDE=""
            CUDA_SECTION=""

            if [ -n "$CUDA_RUNTIME" ]; then
              CUDA_PATH="$(dirname "$CUDA_RUNTIME")"
              echo "Found CUDA path: $CUDA_PATH"
              CUDA_INCLUDE="    - -I$CUDA_PATH"
              CUDA_SECTION="

---

If:
  PathMatch: [.*\\.cu, .*\\.cuh]
CompileFlags:
  Add:
    - --cuda-gpu-arch=sm_86
    - -xcuda
    - --no-cuda-version-check"
            else
              echo "CUDA headers not found under cudatoolkit (this is OK for C++-only usage)" >&2
            fi

            cat > .clangd << EOF
CompileFlags:
  CompilationDatabase: build
  Add:
$(while IFS= read -r line; do
  [ -n "$line" ] && echo "    - -I$line"
done <<< "$CXX_PATHS")
    - -I$root/include
    - -I$root/third_party
$CUDA_INCLUDE
  Remove:
    - -forward-unknown-to-host-compiler
    - --generate-code*$CUDA_SECTION

Diagnostics:
  UnusedIncludes: None
  MissingIncludes: None
EOF

            echo "Generated .clangd"
            echo "Next: generate compile_commands: cmake -B build && ln -sf build/compile_commands.json ."
          '';
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            gnumake
            git
            pkg-config
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart # includes libcudart_static.a
            gccHost
          ];

          LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            cudaPackages.cuda_cudart
          ];

          CPATH = pkgs.lib.makeSearchPath "include" [ ];

          NIX_ENFORCE_NO_NATIVE = "0";
          CC = "${gccHost}/bin/gcc";
          CXX = "${gccHost}/bin/g++";
          CUDACXX = "${cudaPackages.cudatoolkit}/bin/nvcc";

          shellHook = ''
            echo "CUDA toolkit version: ${cudaPackages.cudatoolkit.version}"
          '';
        };

        apps.generate-clangd = {
          type = "app";
          program = "${generateClangd}/bin/generate-clangd";
        };
      }
    );
}
