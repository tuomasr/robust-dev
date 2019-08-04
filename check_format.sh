#!/bin/bash
set -euo pipefail

# Run black.
black *.py

# Run PEP8 checks.
flake8 *.py --max-line-length=99