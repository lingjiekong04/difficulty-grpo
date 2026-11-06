param(
    [Parameter(Mandatory=$true)][string]$CandidateDirectory,
    [Parameter(Mandatory=$true)][string]$CheckoutDirectory
)
# USER-RUN ONLY. The assistant never executes this script.
$ErrorActionPreference='Stop'
$source=(Resolve-Path -LiteralPath $CandidateDirectory).Path
if (-not [IO.Path]::IsPathRooted($CheckoutDirectory)) { throw 'CheckoutDirectory must be absolute.' }
if (Test-Path -LiteralPath $CheckoutDirectory) { throw 'Use a NEW checkout directory; existing work is never overwritten.' }
$manifest=Join-Path $source 'RELEASE_MANIFEST.json'
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw 'Candidate manifest missing.' }
& python -B (Join-Path $source 'tools/verify_release.py')
if ($LASTEXITCODE -ne 0) { throw 'Candidate verification failed; original errors remain above.' }
$repo=Read-Host 'GitHub HTTPS repository URL (create it yourself first)'
if ($repo -notmatch '^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$') { throw 'Use a GitHub HTTPS URL without credentials/query parameters.' }
function Invoke-RepoGit {
    param([string[]]$GitArguments)
    & git -C $CheckoutDirectory @GitArguments
    if ($LASTEXITCODE -ne 0) { throw ('Git failed; full original error is shown above: '+($GitArguments -join ' ')) }
}
& git clone -- $repo $CheckoutDirectory
if ($LASTEXITCODE -ne 0) { throw 'Clone failed; full original Git error is shown above.' }
Invoke-RepoGit -GitArguments @('status','--short')
$dirty=@(& git -C $CheckoutDirectory status --porcelain)
if ($LASTEXITCODE -ne 0) { throw 'Git status failed; original errors remain above.' }
if ($dirty.Count -ne 0) { throw 'Checkout is not clean; no files copied.' }
$paths=@((Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).files.PSObject.Properties.Name)+@('RELEASE_MANIFEST.json')
$existing=@(& git -C $CheckoutDirectory ls-files)
if ($LASTEXITCODE -ne 0) { throw 'Tracked file inspection failed; original errors remain above.' }
$outside=@($existing | Where-Object { $_ -notin $paths })
if ($outside.Count -gt 0) {
    $outside | Write-Host
    throw 'Existing repository contains files outside this strict release. Clone and history are retained; integrate those files manually, without deleting them or resetting history.'
}
if (Test-Path -LiteralPath (Join-Path $CheckoutDirectory 'LICENSE')) {
    if ((Get-FileHash -LiteralPath (Join-Path $CheckoutDirectory 'LICENSE')).Hash -ne (Get-FileHash -LiteralPath (Join-Path $source 'LICENSE')).Hash) { throw 'Existing repository LICENSE differs; resolve licensing manually before this incremental update.' }
}
foreach ($relative in $paths) {
    if ($relative -match '(^/|\\|(^|/)\.\.(/|$))') { throw 'Unsafe manifest path.' }
    $sourceFile=Join-Path $source $relative
    $destination=Join-Path $CheckoutDirectory $relative
    $parent=Split-Path -Parent $destination
    $ancestor=$parent
    while ($ancestor -and $ancestor -ne $CheckoutDirectory) {
        if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Refusing existing symbolic parent directory.' }
        $ancestor=Split-Path -Parent $ancestor
    }
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if (Test-Path -LiteralPath $destination) {
        if ((Get-Item -LiteralPath $destination -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing existing symbolic link.' }
    }
    Copy-Item -LiteralPath $sourceFile -Destination $destination
    Invoke-RepoGit -GitArguments @('add','--',$relative)
}
$staged=@(& git -C $CheckoutDirectory diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Staging inspection failed; original errors remain above.' }
if (@($staged | Where-Object { $_ -notin $paths }).Count -ne 0) { throw 'Unexpected staged path; inspect checkout manually.' }
if ($staged.Count -eq 0) { Write-Host 'No staged changes; no commit or push.'; exit 0 }
Invoke-RepoGit -GitArguments @('diff','--cached','--stat')
Invoke-RepoGit -GitArguments @('status','--short')
$name=Read-Host 'Git author name (repository-local config only)'
$email=Read-Host 'Git author email (fill this yourself; repository-local only)'
if (-not $name.Trim() -or -not $email.Trim()) { throw 'Author identity required.' }
Invoke-RepoGit -GitArguments @('config','--local','user.name',$name)
Invoke-RepoGit -GitArguments @('config','--local','user.email',$email)
if ((Read-Host 'Review the staged changes. Type COMMIT to commit and push') -cne 'COMMIT') { Write-Host 'Stopped; staged files retained for review.'; exit 0 }
Invoke-RepoGit -GitArguments @('commit','-m','Publish audited full-parameter GRPO research delivery')
Invoke-RepoGit -GitArguments @('rev-parse','--verify','HEAD')
Invoke-RepoGit -GitArguments @('push','-u','origin','HEAD')
Write-Host 'Uploaded without force push; prior remote history preserved.'
