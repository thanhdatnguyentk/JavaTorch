@echo off
if exist gradlew.bat (
  gradlew.bat %*
  exit /b %ERRORLEVEL%
)
where gradle >nul 2>&1
if %ERRORLEVEL%==0 (
  gradle %*
  exit /b %ERRORLEVEL%
)
echo Gradle wrapper not found. Please run 'gradle wrapper' locally to generate the wrapper or install Gradle.
exit /b 1
