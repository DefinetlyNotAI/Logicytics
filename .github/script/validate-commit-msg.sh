#!/usr/bin/env sh

set -eu

commit_msg_file="$1"
first_line="$(head -n 1 "$commit_msg_file")"

pattern='^((fixup|squash|amend|reword)! )?(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([A-Za-z0-9._/-]+\))?!?: .+$'

if printf '%s\n' "$first_line" | grep -Eq "$pattern"; then
    exit 0
fi

cat >&2 <<'EOF'

Invalid commit message.

Expected Conventional Commits format:

  <type>: <description>
  <type>(<scope>): <description>
  <type>!: <description>
  <type>(<scope>)!: <description>

Allowed types:

  build
  chore
  ci
  docs
  feat
  fix
  perf
  refactor
  revert
  style
  test

Expected Conventional Commits format:

  <type>: <description>
  <type>(<scope>): <description>
  <type>!: <description>
  <type>(<scope>)!: <description>

Autosquash commits are also accepted:

  fixup! <conventional commit>
  squash! <conventional commit>
  amend! <conventional commit>
  reword! <conventional commit>

EOF

exit 1
