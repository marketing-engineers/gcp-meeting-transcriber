#!/bin/sh
# Configures git to use the tracked .githooks directory for hook enforcement.
set -eu

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

git config core.hooksPath .githooks
chmod +x .githooks/*

echo "Git hooks configured successfully: core.hooksPath = .githooks"
