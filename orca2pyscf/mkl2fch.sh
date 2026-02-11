#!/bin/bash

# Define MOKIT mkl2fch path
MKL2FCH=/home/ubuntu/packages/mokit/bin/mkl2fch
export LD_LIBRARY_PATH=/home/ubuntu/packages/mokit/lib:$LD_LIBRARY_PATH

if [[ ! -x "$MKL2FCH" ]]; then
    echo "Error: mkl2fch executable not found at $MKL2FCH"
    exit 1
fi

# Loop through all .mkl files in the current directory
for mkl in *.mkl; do
    # Check if glob expanded to an existing file
    [[ -e "$mkl" ]] || continue

    # Extract basename (remove .mkl extension)
    basename="${mkl%.mkl}"
    
    echo "Converting $mkl to fch..."
    
    # Usage: mkl2fch input.mkl
    "$MKL2FCH" "$mkl"

    if [[ $? -eq 0 ]]; then
        if [[ -f "${basename}.fch" ]]; then
            echo "Success: Created ${basename}.fch"
        else
            echo "Warning: Command succeeded but ${basename}.fch not found."
        fi
    else
        echo "Error: Failed to convert $mkl"
    fi
done
