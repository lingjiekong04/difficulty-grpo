# 用户手动上传 GitHub

执行位置：本地 Windows PowerShell。不要上传私审包或本机 outputs 父目录，只使用候选目录。

```powershell
$Candidate = Read-Host '候选项目绝对路径'
$Checkout = Read-Host '用于GitHub上传的新checkout绝对路径（必须不存在）'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Candidate 'tools/publish_manual.ps1') -CandidateDirectory $Candidate -CheckoutDirectory $Checkout
if ($LASTEXITCODE -ne 0) { throw '查看上方原始错误；工作区与Git历史均保留' }
```

脚本让你填写实际GitHub HTTPS仓库URL和本仓库Git姓名/邮箱；不会读取登录凭据，不猜代理。GitHub仓库由你创建。clone到新目录，即使远端已有提交也保留历史，不重新init，不reset、不force。核对候选hash→检查checkout干净→逐一复制/暂存白名单文件→检查暂存范围→显示diff统计→本地配置身份→你输入COMMIT后提交→确认HEAD→push。每个原生Git命令后立即检查退出码，原始stderr保留。

如已有仓库存在候选清单之外的文件或不同LICENSE，脚本在写入前停止，保留新clone及全部历史。请在该clone内人工增量整合，不删除旧文件或强推。整合后按真实新增公开文件另建新发布清单与检查记录；不要把旧文件混入本版严格manifest后称本版验证通过。

任何错误都不自动清理clone或撤销暂存。提交/上传只由你手动执行；本轮未运行该脚本，未创建GitHub仓库、未commit/push。
