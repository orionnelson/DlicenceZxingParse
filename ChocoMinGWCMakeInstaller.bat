powershell -Command "& {Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))}"
powershell -Command "& {choco install mingw}"
powershell -Command "& {wget -O cmake.msi https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-windows-x86_64.msi}"
powershell -Command "& {msiexec.exe /I cmake.msi}" 