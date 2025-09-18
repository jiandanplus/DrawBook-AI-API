# 初始化Git仓库并获取子模块的PowerShell脚本

Write-Host "Initializing git repository..." -ForegroundColor Green
git init

Write-Host "Setting up git repository..." -ForegroundColor Green
git config user.name "lora-scripts-user"
git config user.email "user@example.com"

Write-Host "Adding remote origin..." -ForegroundColor Green
git remote add origin https://github.com/hanamizuki-ai/lora-scripts

Write-Host "Fetching submodules information..." -ForegroundColor Green
git submodule init

Write-Host "Updating submodules..." -ForegroundColor Green
git submodule update

Write-Host "Done! You can now run the GUI." -ForegroundColor Green