#!/bin/bash
# Mock kubectl script to intercept manifest application from XPK
REAL_KUBECTL="/usr/bin/kubectl"

if [[ "$*" == *"apply"* || "$*" == *"create"* ]]; then
  # Find if -f is followed by a file path or '-'
  file_path=""
  args=("$@")
  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == "-f" ]]; then
      file_path="${args[i+1]}"
      break
    fi
  done

  if [ -n "$file_path" ]; then
    if [ "$file_path" = "-" ]; then
      cat > generated_manifest.yaml
    else
      cp "$file_path" generated_manifest.yaml
    fi
    echo "Mock kubectl: intercepted apply command and saved manifest to generated_manifest.yaml"
    exit 0
  fi
fi

# Fallback: forward to real kubectl
exec "$REAL_KUBECTL" "$@"
