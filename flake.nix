{
  inputs = {
    # Use unstable so we can access newer CUDA (12.8, 13.x) via cudaPackages_12_8 / cudaPackages_13.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-gl-host.url = "github:numtide/nix-gl-host";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      nix-gl-host,
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
        # cudaPackages = pkgs.cudaPackages_12_9;
        gccHost = pkgs.gcc13;

        nixgl = nix-gl-host.defaultPackage.${system};
        weights = import ./nix/weights.nix { inherit pkgs; };
        mypkgs = import ./nix/packages.nix {
          inherit pkgs cudaPackages gccHost nixgl system weights;
          src = self;
        };

        generateClangd = pkgs.writeShellApplication {
          name = "generate-clangd";
          runtimeInputs = with pkgs; [
            git
            findutils
            gnugrep
            gnused
            coreutils
            gccHost
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
            : "\$\{CXX:=c++\}"
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
    - --cuda-path=${cudaPackages.cudatoolkit}
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
          '';
        };
      in
      {
        devShells.default = pkgs.mkShellNoCC {
          packages = with pkgs; [
            cmake
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            gccHost
            nixgl
            go-task
          ];
          # LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath packages;

          # Pin host and CUDA compilers for CMake and nvcc
          #  This doesnt appear strictly neccessary rn but it prevents nvcc from
          #    using a different compiler than what we explictly request and have
          #    available in our env (dont know where its getting v14 from when we
          #    dont have this btw).
          CC = "${gccHost}/bin/gcc";
          CXX = "${gccHost}/bin/g++";
          CUDACXX = "${cudaPackages.cudatoolkit}/bin/nvcc";
          # Convenient sets for downstream tooling, again not strictly neccessary for
          #  compiling+running kernels (but I, for one, do use CUDA_HOME)
          # Who knows what is affected by the other sets but yeah idk its solved things before.
          # LIBRARY_PATH = pkgs.lib.makeLibraryPath [
          #   cudaPackages.cuda_cudart
          # ];
          #
          # CPATH = pkgs.lib.makeSearchPath "include" [ ];
          CUDA_HOME = cudaPackages.cudatoolkit;
          # CUDAToolkit_ROOT = cudaPackages.cudatoolkit;
          # CUDA_PATH = cudaPackages.cudatoolkit;

          shellHook = ''
            # this is magic never touch this lol
            export LD_LIBRARY_PATH=$(nixglhost -p):$LD_LIBRARY_PATH
            echo "NIX_ENFORCE_NO_NATIVE=$NIX_ENFORCE_NO_NATIVE"
          '';
        };

        apps.generate-clangd = {
          type = "app";
          program = "${generateClangd}/bin/generate-clangd";
        };

        packages = {
          inference = mypkgs.inference.build;
          inherit weights;
        };

        apps = {
          inference = {
            type ="app";
            program = "${mypkgs.inference.run}/bin/inference";
          };
        };
      }
    );
}
