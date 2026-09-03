# Windows MinGW-w64 Setup

This guide prepares a Windows development environment for building the native C EdgeXPU-LLM MVP.

## Required Tools

- CMake for Windows
- MSYS2 with MinGW-w64 UCRT64 toolchain
- GCC, Ninja, and MinGW make from MSYS2

## Install CMake

Download the Windows x86_64 CMake package from:

https://cmake.org/download/

After extracting it, the executable should look like this:

```text
D:\cmake-4.4.3-windows-x86_64\cmake-4.4.3-windows-x86_64\bin\cmake.exe
```

Verify from PowerShell:

```powershell
& "D:\cmake-4.4.3-windows-x86_64\cmake-4.4.3-windows-x86_64\bin\cmake.exe" --version
```

## Install MSYS2

Download and install MSYS2 from:

https://www.msys2.org

Open the **MSYS2 UCRT64** terminal. Do not use the plain **MSYS2 MSYS** terminal for this project.

Update packages:

```bash
pacman -Syu
```

If MSYS2 asks you to close the terminal, close it, reopen **MSYS2 UCRT64**, then run:

```bash
pacman -Syu
```

Install the native build tools:

```bash
pacman -S --needed mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-ninja
```

## Configure Windows PATH

Add this directory to the Windows user PATH:

```text
C:\msys64\ucrt64\bin
```

After updating PATH, restart Cursor and PowerShell.

Verify the tools from PowerShell:

```powershell
gcc --version
mingw32-make --version
ninja --version
```

## Build EdgeXPU-LLM

From the repository root:

```powershell
& "D:\cmake-4.4.3-windows-x86_64\cmake-4.4.3-windows-x86_64\bin\cmake.exe" -S . -B build -G Ninja
& "D:\cmake-4.4.3-windows-x86_64\cmake-4.4.3-windows-x86_64\bin\cmake.exe" --build build
```

If CMake is already in PATH, the shorter form is:

```powershell
cmake -S . -B build -G Ninja
cmake --build build
```

## Run Basic Commands

Inspect local capabilities:

```powershell
.\build\edgexpu.exe capabilities
```

Inspect a model manifest:

```powershell
.\build\edgexpu.exe inspect-manifest examples\models\smollm2-135m\model.manifest.json
```

Start the local API server:

```powershell
.\build\edgexpu.exe serve examples\models\smollm2-135m\model.manifest.json 8000
```

## Common Problems

If CMake reports `CMAKE_C_COMPILER not set`, Windows cannot find GCC. Confirm that `C:\msys64\ucrt64\bin` is in PATH and that `gcc --version` works in a new PowerShell.

If CMake tries to use `NMake Makefiles` and fails with `nmake` not found, pass `-G Ninja` when configuring:

```powershell
cmake -S . -B build -G Ninja
```

If `edgexpu benchmark` or `edgexpu serve` fails because no CPU backend is available, install or build a local PowerInfer/llama.cpp-compatible binary such as `llama-cli` and make sure it is on PATH.
