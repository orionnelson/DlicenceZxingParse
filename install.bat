powershell -Command "& {python -m venv .}"
powershell -Command "& {.\DlicenceZxingParse\Scripts\activate.bat}"
powershell -Command "& {cd DlicenceZxingParse}"
powershell -Command "& {.\ChocoMinGWCMakeInstaller.bat}"