#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
# The `set -e` is crucial here for the `||` trick to work as expected.
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
# Reset code to return to the default terminal color
NC='\033[0m' # No Color

# Variable to hold the captured stderr
CAPTURED_STDERR=""
# Variable to hold the captured exit code
CAPTURED_EXIT_CODE=0

CURRENT_SCRIPT=""

# Define the error handling function
handle_error() {
  echo -e "${RED}[fail]${NC}:$BASH_COMMAND:$CAPTURED_STDERR ($CAPTURED_EXIT_CODE)"
}

handle_success(){
  echo -e "${GREEN}[success]${NC}:$CURRENT_SCRIPT:$CAPTURED_STDOUT ($CAPTURED_EXIT_CODE)"
}

# Define a function to execute a command and capture its output
# Using `local` prevents these temp variables from leaking
capture_and_run() {
  local stderr_file=$(mktemp)
  local stdout_file=$(mktemp)
  # Execute the command, redirecting stderr to the temp file
  # The `|| true` prevents `set -e` from immediately exiting
  ("$@" > "$stdout_file" 2> "$stderr_file") || true

  # Save the command's exit code
  CAPTURED_EXIT_CODE=$?
  # Capture the stderr from the temporary file
  CAPTURED_STDOUT=$(cat "$stdout_file")
  CAPTURED_STDERR=$(cat "$stderr_file")
  # Remove the temporary file
  rm "$stdout_file" "$stderr_file"
}

# Widely accepted as the most robust way to get the absolute path containing this file
SCRIPT_DIR_ABS="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

# It's a bash-specific feature and is best for handling basic relative paths. 
# It does not resolve symlinks or nested relative paths.
SCRIPT_DIR_RELATIVE="${BASH_SOURCE[0]%/*}"

# Uses standard POSIX commands and does not change the current working directory because it runs in a subshell (...). 
# It will correctly resolve relative paths but will still give the directory of the symlink, not the target script.
#SCRIPT_DIR_ABS_V2=$( (cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd) )

# Fails if run with `source`
#SCRIPT_DIR_ABS_V3=$(dirname "$(realpath "$0")")

#echo $SCRIPT_DIR_ABS
#echo $SCRIPT_DIR_RELATIVE
#echo $SCRIPT_DIR_ABS_V2
#echo $SCRIPT_DIR_ABS_V3

FILE="$SCRIPT_DIR_RELATIVE/verify_example_operations_active.lst"

if [[ ! -f "$FILE" ]]; then
  echo "Error: File '$FILE' not Found"
  exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do

 CURRENT_SCRIPT="$line"
 capture_and_run $CURRENT_SCRIPT

 # Check the captured exit code and manually call the handler
 if [ "$CAPTURED_EXIT_CODE" -ne 0 ]; then
   handle_error
 else
   handle_success
 fi

done < "$FILE"
