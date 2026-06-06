param(
    [string]$OutputZip = "$HOME\Downloads\DoAnTotNghiep_code_colab.zip"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("kws_colab_pkg_" + [System.Guid]::NewGuid().ToString("N"))
$PackageRoot = Join-Path $TempRoot "DoAnTotNghiep"

New-Item -ItemType Directory -Force -Path $PackageRoot | Out-Null

$excludeDirs = @(
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "data\enroll_profiles",
    "data\gsc_v2",
    "data\mswc_en",
    "data\test",
    "data\__pycache__",
    "checkpoints",
    "results",
    "logs",
    "logs_colab",
    "server\DoAnTotNghiep_output-20260522T014622Z-3-001",
    "server\final_kws_artifacts_package"
)

$excludeFiles = @(
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.tar.gz",
    "*.zip",
    "*.wav",
    "*.opus",
    "*.flac",
    "*.mp3",
    "*.mp4",
    "*.avi",
    "*.npy",
    "*.npz"
)

try {
    $robocopyArgs = @($RepoRoot, $PackageRoot, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS")
    foreach ($dir in $excludeDirs) {
        $robocopyArgs += "/XD"
        $robocopyArgs += (Join-Path $RepoRoot $dir)
    }
    foreach ($file in $excludeFiles) {
        $robocopyArgs += "/XF"
        $robocopyArgs += $file
    }

    & robocopy @robocopyArgs | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code $LASTEXITCODE"
    }

    if (Test-Path $OutputZip) {
        Remove-Item -LiteralPath $OutputZip -Force
    }
    Compress-Archive -Path $PackageRoot -DestinationPath $OutputZip -Force
    Write-Host "Created: $OutputZip"
}
finally {
    if (Test-Path $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
    }
}
