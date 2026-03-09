#!/usr/bin/env sh
if [ -f "./gradlew" ] && [ "$(basename "$0")" = "gradlew" ]; then
  exec "./gradlew" "$@"
fi
if command -v gradle >/dev/null 2>&1; then
  exec gradle "$@"
fi
echo "Gradle wrapper not found. Please run 'gradle wrapper' locally to generate the wrapper or install Gradle."
exit 1
