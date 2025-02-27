#!/bin/bash

# Make run.sh fail if a subcommand fails
set -e
set -o pipefail

# Command line arguments?
if [[ -n "$1" ]]; then
	script="$1"

	# Run script provided in command line
	python3 $script
else
	# Run all .py scripts in directory
	for script in *.py; do
		# Fancy printing
		printcmd="Now running $script"
		varlength=${#printcmd}

		printf '%*s\n' "$varlength" '' | tr ' ' '-'
		echo $printcmd
		echo
		python3 $script
	done
fi
