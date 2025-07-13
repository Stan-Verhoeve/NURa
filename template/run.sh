#!/bin/bash

# Start timer
start_time=$(date +%s.%N)

# Check if folder for data exists
if [[ ! -d "data" ]]; then
    # If not, create it
    echo "Creating 'data' directory..."
    mkdir data

    # And download data files to it
    echo "Downloading satellite data..."
    wget -P ./data https://home.strw.leidenuniv.nl/~daalen/Handin_files/galaxy_data.txt
else
    echo "'data' directory already exists. Now checking for missing files..."
    
    if [[ ! -f "./data/galaxy_data.txt" ]]; then
        echo "galaxy_data.txt is missing. Downloading..."
        wget -P ./data https://home.strw.leidenuniv.nl/~daalen/Handin_files/galaxy_data.txt
    else
        echo "galaxy_data.txt already exists."
    fi

    if [[ ! -f "./data/DMO_a0.1_256.hdf5" ]]; then
        echo "DMO_a0.1_256.hdf5 is missing. Too large to download; move here yourself..."
    else
        echo "DMO_a0.1_256.hdf5 already exists."
    fi
fi

# Check if 'galaxy_data.txt' exists
if [[ ! -f "data/galaxy_data.txt" ]]; then
	echo "galaxy_data.txt does not exist. Grabbing..."
    wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/galaxy_data.txt ./data
fi

# Check if folder for figures exists
if [[ ! -d "figures" ]]; then
	# If not, create it
	echo "Creating 'figures' directory..."
	mkdir figures
    mkdir figures/movie
else
	# If so, clear it
	echo "'figures' directory already exists. Now clearing..."
	rm -rf figures/*
    mkdir figures/movie
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

# Make files
echo "Now making files"
make

# Do we have command line arguments?
if [[ -n "$1" ]]; then
	script="$1"
    
    # Extract extension, if any
    if [[ "$script" == *.* ]]; then
        extension="${script##*.}"
        base="${script%.*}"
    else
        extension=""
        base="$script"
    fi
	
    # Fancy printing
	printcmd="Now running $script"
	varlength=${#printcmd}
    
	printf '%*s\n' "$varlength" '' | tr ' ' '-'
	echo $printcmd
	echo
    
    if [[ "$extension" == "py" ]]; then
        # Run script provided in command line
        python3 $script
    elif [[ "$extension" == "c" ]]; then
        if [[ -x "$base" ]]; then
            ./"$base"
        else
            echo "Executable $base not found or not executable."
        fi
    elif [[ -x "$script" ]]; then
        ./"$script"
    else
        echo "Unrecognized or non-executable file: $script"
    fi
    
    # Clean up after ourselves
    make clean

    # Stop timer
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)

    # Round to 3 decimal places
    elapsed_rounded=$(printf "%.3f" "$elapsed")

    # Compute minutes
    elapsed_minutes=$(echo "$elapsed / 60" | bc -l)
    elapsed_minutes_rounded=$(printf "%.3f" "$elapsed_minutes")

    printcmd="Total execution time: $elapsed_rounded seconds, or $elapsed_minutes_rounded minutes"
    varlength=${#printcmd}
    printf '%*s\n' "$varlength" '' | tr ' ' '-'
    echo $printcmd
    
    # Exit
	exit 0
else
    # Run all compiled .c binaries first
    for cfile in *.c; do
        base="${cfile%.c}"
        if [[ -x "$base" ]]; then  # check if binary exists and is executable
            printcmd="Now running $base (compiled from $cfile)"
            varlength=${#printcmd}
            outfile="OUT/${base}.txt"

            printf '%*s\n' "$varlength" '' | tr ' ' '-'
            echo "$printcmd"
            echo
            ./"$base" > "$outfile"
        fi
    done

    # Then run all .py scripts
    for script in *.py; do
        printcmd="Now running $script"
        varlength=${#printcmd}
        outfile="OUT/${script%.py}.txt"

        printf '%*s\n' "$varlength" '' | tr ' ' '-'
        echo "$printcmd"
        echo
        python3 "$script" > "$outfile"
    done
fi

# Clean after making
make clean

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

# Stop timer
end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)

# Round to 3 decimal places
elapsed_rounded=$(printf "%.3f" "$elapsed")

# Compute minutes
elapsed_minutes=$(echo "$elapsed / 60" | bc -l)
elapsed_minutes_rounded=$(printf "%.3f" "$elapsed_minutes")

printcmd="Total execution time: $elapsed_rounded seconds, or $elapsed_minutes_rounded minutes"
varlength=${#printcmd}
printf '%*s\n' "$varlength" '' | tr ' ' '-'
echo $printcmd
