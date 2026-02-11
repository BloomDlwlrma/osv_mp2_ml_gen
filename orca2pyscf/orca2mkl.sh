#!/bin/bash

# Define ORCA paths (adjust if needed or rely on module load)
export PATH=/share/bin/orca_6_0_0:$PATH
export LD_LIBRARY_PATH=/share/bin/orca_6_0_0:$LD_LIBRARY_PATH

# Loop through all .gbw files in the current directory
for gbw in *.gbw; do
    # Check if glob expanded to an existing file
    [[ -e "$gbw" ]] || continue

    # Extract basename (remove .gbw extension)
    basename="${gbw%.gbw}"
    
    # Define corresponding output file name
    out="${basename}.out"

    # Check if both .gbw and .out exist
    if [[ -f "$gbw" && -f "$out" ]]; then
        echo "Found pair: $gbw and $out"
        echo "Converting to MKL..."
        
        # Usage: orca_2mkl basename -mkl
        # Takes the basename (without extension) and assumes .gbw exists
        orca_2mkl "$basename" -mkl

        if [[ $? -eq 0 ]]; then
            echo "Success: Created ${basename}.mkl"
        else
            echo "Error: Failed to create ${basename}.mkl"
        fi
    else
        # Optional: notify if .out is missing for a .gbw
        if [[ ! -f "$out" ]]; then
            echo "Skipping $basename: $out not found."
        fi
    fi
done

# Loop through all .loc files in the current directory
for loc in *.loc; do
    # Check if glob expanded to an existing file
    [[ -e "$loc" ]] || continue

    # Extract basename (remove .loc extension)
    basename="${loc%.loc}"
    
    echo "Found loc file: $loc"
    
    # Create temporary gbw for conversion to achieve _loc.mkl naming
    cp "$loc" "${basename}_loc.gbw"
    
    echo "Converting to MKL..."
    # Usage: orca_2mkl basename -mkl
    orca_2mkl "${basename}_loc" -mkl

    if [[ $? -eq 0 ]]; then
        echo "Success: Created ${basename}_loc.mkl"
        rm "${basename}_loc.gbw"
    else
        echo "Error: Failed to create ${basename}_loc.mkl"
    fi
done
