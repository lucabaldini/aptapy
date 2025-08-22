#!/bin/bash

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Base package root. All the other relevant folders are relative to this location.
export APTAPY_ROOT=$SETUP_DIR

# Add the root folder to the $PYTHONPATH environmental variable.
export PYTHONPATH=$APTAPY_ROOT:$PYTHONPATH
