param(
    [string]$OutputZip = "$HOME\Downloads\DoAnTotNghiep_code_colab_POSIX.zip"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

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
    "server\final_kws_artifacts_package",
    ".codex_tmp"
)

$excludeFilePatterns = @(
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

function Is-ExcludedPath {
    param([string]$FullPath)
    foreach ($dir in $excludeDirs) {
        $dirPath = Join-Path $RepoRoot $dir
        if ($FullPath.Equals($dirPath, [System.StringComparison]::OrdinalIgnoreCase) -or
            $FullPath.StartsWith($dirPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
            return $true
        }
    }
    return $false
}

function Matches-FilePattern {
    param([string]$Name)
    foreach ($pattern in $excludeFilePatterns) {
        if ($Name -like $pattern) {
            return $true
        }
    }
    return $false
}

function Add-FilesToZip {
    param(
        [System.IO.Compression.ZipArchive]$Zip,
        [string]$Directory
    )

    if (Is-ExcludedPath $Directory) {
        return
    }

    Get-ChildItem -LiteralPath $Directory -Force -File -ErrorAction SilentlyContinue | ForEach-Object {
        if (Matches-FilePattern $_.Name) {
            return
        }
        $rel = $_.FullName.Substring($RepoRoot.Length).TrimStart("\", "/")
        $entryName = ("DoAnTotNghiep/" + ($rel -replace "\\", "/"))
        try {
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $Zip,
                $_.FullName,
                $entryName,
                [System.IO.Compression.CompressionLevel]::Optimal
            ) | Out-Null
        }
        catch {
            Write-Warning "Skipping locked/unreadable file: $rel ($($_.Exception.Message))"
        }
    }

    Get-ChildItem -LiteralPath $Directory -Force -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        if (-not (Is-ExcludedPath $_.FullName)) {
            Add-FilesToZip -Zip $Zip -Directory $_.FullName
        }
    }
}

if (Test-Path $OutputZip) {
    Remove-Item -LiteralPath $OutputZip -Force
}

$outDir = Split-Path -Parent $OutputZip
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$fs = [System.IO.File]::Open($OutputZip, [System.IO.FileMode]::CreateNew)
try {
    $zip = New-Object System.IO.Compression.ZipArchive($fs, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Add-FilesToZip -Zip $zip -Directory $RepoRoot
    }
    finally {
        $zip.Dispose()
    }
}
finally {
    $fs.Dispose()
}

Write-Host "Created POSIX zip: $OutputZip"
