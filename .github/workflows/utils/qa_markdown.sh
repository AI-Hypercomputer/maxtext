#!/usr/bin/env bash

set -euo pipefail

g_coverage=0
g_pyink=0
g_pylint=0
g_pytype=0
g_codespell=0
printf 'filename | coverage | pyink | pylint | pytype | codespell\n---------|----------|-------|--------|--------|----------\n' ;
# coverage run -m pytest --junitxml=reports/junit/junit.xml
coverage='⚪';
while IFS= read -r -d '' p; do
  if python3 -m pylint --disable C0114,R0401,R0917,W0201,W0613 "$p" &>/dev/null; then
    pylint='🟢';
  else
    pylint='🔴';
    g_pylint=1
  fi;

  if pyink MaxText --check --diff --color --pyink-indentation=2 --line-length=125 "$p" &>/dev/null; then
    pyink='🟢';
  else
    pyink='🔴';
    g_pyink=1
  fi

  if pytype --jobs auto --keep-going "$p" &>/dev/null; then
    pytype='🟢';
  else
    pytype='🔴';
    g_pytype=1
  fi

  if codespell -w --skip="*.txt,pylintrc,.*,assets/*" -L ND,nd,sems,TE,ROUGE,rouge "$p" &>/dev/null; then
    codespell='🟢';
  else
    codespell='🔴';
    g_codespell=1
  fi

  printf '%s' "${p}" ;
  printf '| %s ' "${coverage}" "${pyink}" "${pylint}" "${pytype}" "${codespell}";
  printf '\n';
done< <(git diff -z --name-only HEAD main -- '*.py' '*.ipynb')

if [ "${g_pylint}" -eq 1 ]; then
  # shellcheck disable=SC2016
  printf '\nOn your changes you need to run:\n```sh\n%s\n```\n' \
    'python3 -m pylint --disable C0114,R0401,R0917,W0201,W0613'
fi

if [ "${g_pyink}" -eq 1 ]; then
  # shellcheck disable=SC2016
  printf '\nOn your changes you need to run:\n```sh\n%s\n```\n' 'pyink --pyink-indentation=2 --line-length=122'
fi

if [ "${g_coverage}" -eq 1 ]; then
  # shellcheck disable=SC2016
  printf '\nTODO: auto run and parse:\n```sh\n%s\n```\n' ' coverage run -m pytest --junitxml=reports/junit/junit.xml'
fi

if [ "${g_pytype}" -eq 1 ]; then
  # shellcheck disable=SC2016
  printf '\nOn your changes you need to run:\n```sh\n%s\n```\n' 'pytype --jobs auto --keep-going'
fi

if [ "${g_codespell}" -eq 1 ]; then
  # shellcheck disable=SC2016
  printf '\nOn your changes you need to run:\n```sh\n%s\n```\n' \
    'codespell -w --skip="*.txt,pylintrc,.*,assets/*" -L ND,nd,sems,TE,ROUGE,rouge'
fi
