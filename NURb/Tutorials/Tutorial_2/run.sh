#!/bin/bash

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

# Check if folder for txt exists
if [[ ! -d "OUT" ]]; then
	# If not, create it
	echo "Creating 'OUT' directory..."
	mkdir OUT
else
	# If so, clear it
	echo "'OUT' directory already exists. Now clearing..."
	rm -rf OUT/*
fi

# Do we have command line arguments?
if [[ -n "$1" ]]; then
	script="$1"

	# Fancy printing
	printcmd="Now running $script"
	varlength=${#printcmd}
	
	printf '%*s\n' "$varlength" '' | tr ' ' '-'
	echo $printcmd
	echo

	# Run script provided in command line
	python3 $script

	# Exit
	exit 0
else
	# Run all .py scripts in directory
	for script in *.py; do
		# Fancy printing
		printcmd="Now running $script"
		varlength=${#printcmd}
		
		# txt to write output to
		outfile="OUT/${script%.py}.txt"

		printf '%*s\n' "$varlength" '' | tr ' ' '-'
		echo $printcmd
		echo
		python3 $script > $outfile
	done
fi

# Check if latex folder exists
if [[ ! -d "latex" ]]; then
	echo "No 'latex' direcotry found. Skipping compilation..."
else
	# If no latex_out folder, create it
	if [[ ! -d "latex/OUT" ]]; then
		echo "Creating 'OUT' directory..."
		mkdir latex/OUT
	else
		# else clear it
		echo "'OUT' directory already exists. Now clearing..."
		rm -rf latex/OUT/*
	fi
	
	# Check if there are any .tex files
	if [[ -f "latex/main.tex" ]]; then
		echo "Creating PDFs from LaTeX files"
		
		cd latex
		pdflatex -output-dir OUT main.tex
		bibtex OUT/main
		pdflatex -output-dir OUT main.tex
		pdflatex -output-dir OUT main.tex
		cd ..
	else
		echo "No main.tex file found. Skipping compilation..."
	fi
fi
