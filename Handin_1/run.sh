#!/bin/bash

# Make run.sh fail if a subcommand fails
set -e
set -o pipefail

# Check if folder for figures exists
if [[ ! -d "figures" ]]; then
	# If not, create it
	echo "Creating 'figures' directory..."
	mkdir figures
else
	# If so, clear it
	echo "'figures' directory already exists. Now clearing..."
	rm -rf figures/*
fi

# Do we have command line arguments?
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

# Check if latex folder exists
if [[ ! -d "latex" ]]; then
	echo "No 'latex' direcotry found. Skipping compilation..."
else
	# If no latex_out folder, create it
	if [[ ! -d "latex_out" ]]; then
		echo "Creating 'latex_out' directory..."
		mkdir latex_out
	fi
	
	# Check if there are any .tex files
	if compgen -G "latex/*.tex" > /dev/null; then
		echo "Creating PDFs from LaTeX files"
	
		for file in "latex/*.tex"; do
			pdflatex -output-dir latex_out $file
		done
	else
		echo "No .tex files found. Skipping compilation..."
	fi
fi
