param(
  [switch]$Clean,
  [switch]$WithDA3,
  [switch]$SetupBuildTools
)

$ErrorActionPreference = "Stop"

# Support --clean/--withda3/--setupbuildtools style arguments.
$unknownArgs = @()
foreach ($arg in $args) {
  switch -Regex ($arg) {
    "^--clean$" { $Clean = $true; continue }
    "^--withda3$" { $WithDA3 = $true; continue }
    "^--setupbuildtools$" { $SetupBuildTools = $true; continue }
    default { $unknownArgs += $arg }
  }
}
if ($unknownArgs.Count -gt 0) {
  Write-Host "Unknown arguments: $($unknownArgs -join ' ')" -ForegroundColor Yellow
  Write-Host "Valid options: -Clean, -WithDA3, -SetupBuildTools (or --clean/--withda3/--setupbuildtools)"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv"
$pipIndex = "https://download.pytorch.org/whl/cu128"
$programFiles64 = $env:ProgramW6432
if (-not $programFiles64) {
  $programFiles64 = $env:ProgramFiles
}
$defaultCudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$defaultCudaHome64 = Join-Path $programFiles64 "NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$defaultCudaHomeX86 = Join-Path ${env:ProgramFiles(x86)} "NVIDIA GPU Computing Toolkit\CUDA\v12.8"

Write-Host "Repo root: $repoRoot"
Write-Host "Venv:      $venvPath"

if ($Clean -and (Test-Path $venvPath)) {
  if ($env:VIRTUAL_ENV -and (Resolve-Path $env:VIRTUAL_ENV) -eq (Resolve-Path $venvPath)) {
    if (Get-Command deactivate -ErrorAction SilentlyContinue) {
      Write-Host "Deactivating current venv..."
      deactivate
    }
  }
  Write-Host "Cleaning existing venv..."
  try {
    Remove-Item -Recurse -Force $venvPath -ErrorAction Stop
  } catch {
    Write-Host "Failed to remove $venvPath. Close any Python processes and run from a non-venv shell." -ForegroundColor Yellow
    throw
  }
}

function Initialize-VSDevEnv {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) {
    $vswhere = Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe"
  }

  $candidates = @()
  if (Test-Path $vswhere) {
    $instances = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json | ConvertFrom-Json
    foreach ($inst in $instances) {
      $vcvars = Join-Path $inst.installationPath "VC\Auxiliary\Build\vcvars64.bat"
      if (Test-Path $vcvars) {
        $candidates += [pscustomobject]@{
          Version = $inst.installationVersion
          Path = $vcvars
        }
      }
    }
  }

  if (-not $candidates) {
    $roots = @(
      "C:\Program Files (x86)\Microsoft Visual Studio",
      "C:\Program Files\Microsoft Visual Studio"
    ) | Where-Object { Test-Path $_ }
    foreach ($root in $roots) {
      Get-ChildItem -Path $root -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
        ForEach-Object {
          $candidates += [pscustomobject]@{
            Version = "unknown"
            Path = $_.FullName
          }
        }
    }
  }

  if (-not $candidates) {
    Write-Host "No Visual Studio Build Tools found. Install Desktop development with C++." -ForegroundColor Yellow
    return $false
  }

  Write-Host "Found Visual Studio vcvars64.bat:"
  for ($i = 0; $i -lt $candidates.Count; $i++) {
    $label = $candidates[$i].Version
    Write-Host "[$i] $($candidates[$i].Path) (version: $label)"
  }

  $selected = $candidates | Sort-Object Version | Select-Object -Last 1
  Write-Host ""
  Write-Host "Default selection:" -NoNewline
  Write-Host " $($selected.Path)"
  Read-Host "Press Enter to initialize the VS build environment and continue"

  $envLines = cmd /c "`"$($selected.Path)`" && set"
  foreach ($line in $envLines) {
    if ($line -match "^(.*?)=(.*)$") {
      $name = $matches[1]
      $value = $matches[2]
      Set-Item -Path "Env:$name" -Value $value
    }
  }

  $env:DISTUTILS_USE_SDK = "1"
  Write-Host "VS build environment initialized. Verify with: where cl"
  return $true
}

function Ensure-CompilerOnPath {
  $hasCl = Get-Command cl.exe -ErrorAction SilentlyContinue
  if ($hasCl) {
    return $true
  }
  return Initialize-VSDevEnv
}

function Resolve-CudaHome {
  if ($env:CUDA_HOME -and (Test-Path $env:CUDA_HOME)) {
    return $env:CUDA_HOME
  }
  if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
    return $env:CUDA_PATH
  }
  if (Test-Path $defaultCudaHome) {
    return $defaultCudaHome
  }
  return $null
}

function List-CudaInstallations {
  $roots = @(
    (Join-Path $programFiles64 "NVIDIA GPU Computing Toolkit\CUDA"),
    (Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"),
    (Join-Path ${env:ProgramFiles(x86)} "NVIDIA GPU Computing Toolkit\CUDA")
  ) | Where-Object { $_ -and (Test-Path $_) }

  $found = @()
  foreach ($root in $roots) {
    Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match "^v\\d+\\.\\d+$" } |
      ForEach-Object { $found += $_.FullName }
  }

  $found = $found | Sort-Object -Unique
  if ($found.Count -gt 0) {
    Write-Host "Found CUDA installations:"
    for ($i = 0; $i -lt $found.Count; $i++) {
      Write-Host "[$i] $($found[$i])"
    }
  } else {
    Write-Host "No CUDA installations found under standard locations." -ForegroundColor Yellow
  }
  return $found
}

function Ensure-Cuda128 {
  $candidates = List-CudaInstallations
  $defaultFound = @(
    $defaultCudaHome64,
    $defaultCudaHome,
    $defaultCudaHomeX86
  ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

  if ($defaultFound) {
    $cudaHome = $defaultFound
  } else {
    $cudaHome = Resolve-CudaHome
    if (-not $cudaHome -and $candidates) {
      $cuda128 = $candidates | Where-Object { $_ -match "v12\.8$" } | Select-Object -First 1
      if ($cuda128) {
        $cudaHome = $cuda128
      }
    }
  }
  if (-not $cudaHome) {
    Write-Host "CUDA 12.8 not found. Install it at $defaultCudaHome or set CUDA_HOME." -ForegroundColor Yellow
    exit 1
  }
  $env:CUDA_HOME = $cudaHome
  $env:CUDA_PATH = $cudaHome
  $env:Path = "$cudaHome\bin;$cudaHome\libnvvp;$env:Path"

  $nvcc = Join-Path $cudaHome "bin\nvcc.exe"
  if (-not (Test-Path $nvcc)) {
    Write-Host "CUDA 12.8 nvcc not found at $nvcc. Install CUDA 12.8." -ForegroundColor Yellow
    exit 1
  }

  $ver = & $nvcc --version 2>$null
  if (-not ($ver -match "release 12\.8")) {
    Write-Host "nvcc is not CUDA 12.8. Detected:" -ForegroundColor Yellow
    Write-Host ($ver -join "`n")
    exit 1
  }
  Write-Host "CUDA 12.8 detected: $cudaHome"
}

function Ensure-Colmap {
  $colmapRoot = Join-Path $repoRoot "tools\colmap"
  $colmapBat = Join-Path $colmapRoot "COLMAP.bat"
  $colmapExe = Join-Path $colmapRoot "colmap.exe"
  if ((Test-Path $colmapBat) -or (Test-Path $colmapExe)) {
    Write-Host "COLMAP found: $colmapRoot"
    return $true
  }
  Write-Host "COLMAP not found under $colmapRoot. Copy a COLMAP build into tools\\colmap or set it in the UI." -ForegroundColor Yellow
  return $false
}

function Prebuild-Gsplat {
  Ensure-Cuda128

  if (-not $env:TORCH_CUDA_ARCH_LIST) {
    $env:TORCH_CUDA_ARCH_LIST = "8.6"
  }
  Write-Host "Prebuilding gsplat CUDA extension (this may take a few minutes)..."
  & python -c "from gsplat.cuda import _backend as b; print('gsplat CUDA ready:', b._C is not None)"
  $env:TORCH_CUDA_ARCH_LIST = $null
}

if (-not (Test-Path $venvPath)) {
  Write-Host "Checking Visual Studio Build Tools..."
  Ensure-CompilerOnPath | Out-Null
  $env:DISTUTILS_USE_SDK = "1"
  Write-Host "Checking CUDA 12.8..."
  Ensure-Cuda128
  Write-Host "Checking COLMAP..."
  Ensure-Colmap | Out-Null

  Write-Host "Creating venv..."
  & python -m venv $venvPath

  Write-Host "Activating venv..."
  & "$venvPath\Scripts\Activate.ps1"

  Write-Host "Upgrading pip..."
  & python -m pip install --upgrade pip

  Write-Host "Installing build tools for gsplat..."
  & pip install "setuptools" "wheel" "ninja" "numpy"

  if (Test-Path (Join-Path $repoRoot ".gitmodules")) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
      Write-Host "Updating git submodules..."
      & git -C $repoRoot submodule update --init --recursive
    } else {
      Write-Host "git not found; skipping submodule update." -ForegroundColor Yellow
    }
  } else {
    Write-Host "No .gitmodules found; skipping submodule update."
  }

  Write-Host "Installing PyTorch CUDA 12.8 build..."
  & pip install --extra-index-url $pipIndex "torch==2.9.1+cu128"

  $constraints = Join-Path $env:TEMP "nullsplats_constraints.txt"
  "torch==2.9.1+cu128" | Set-Content -Encoding ASCII $constraints

  Write-Host "Building gsplat from source (CUDA extension)..."
  & pip install --no-deps --no-build-isolation --no-binary=gsplat --force-reinstall "gsplat==1.5.3"
  Prebuild-Gsplat

  Write-Host "Installing requirements..."
  & pip install -r "$repoRoot\requirements.txt" -c $constraints --extra-index-url $pipIndex --no-binary=gsplat --no-build-isolation
  Remove-Item $constraints -ErrorAction SilentlyContinue

  Write-Host "Installing Depth Anything 3..."
  if (Test-Path (Join-Path $repoRoot "tools\depth-anything-3")) {
    & pip install -e "$repoRoot\tools\depth-anything-3"
  } else {
    & pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3"
  }

  Write-Host "Setup complete. You can now run: python main.py"
} else {
  Write-Host "Venv already exists. Activating only..."
  & "$venvPath\Scripts\Activate.ps1"

  Write-Host "Checking Visual Studio Build Tools..."
  Ensure-CompilerOnPath | Out-Null
  Write-Host "Checking CUDA 12.8..."
  Ensure-Cuda128
  Write-Host "Checking COLMAP..."
  Ensure-Colmap | Out-Null

  Write-Host "Installing Depth Anything 3..."
  if (Test-Path (Join-Path $repoRoot "tools\depth-anything-3")) {
    & pip install -e "$repoRoot\tools\depth-anything-3"
  } else {
    & pip install "git+https://github.com/ByteDance-Seed/Depth-Anything-3"
  }

  Write-Host "Venv activated. You can now run: python main.py"
}
