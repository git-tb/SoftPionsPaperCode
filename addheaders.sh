#!/bin/bash

find . -type f -name "freezeout.csv" | while read -r file; do    
    # Create a temp file with the new line + original content
    tmpfile=$(mktemp)
    echo "alpha,tau,r,Dtau,Dr" > "$tmpfile"
    cat "$file" >> "$tmpfile"    
    # Move the temp file back to the original file
    mv "$tmpfile" "$file"
done

find . -type f -name "fluidumdata.csv" | while read -r file; do
    tmpfile=$(mktemp)
    echo "PT [GEV],(1/Nev)*(1/(2*PI*PT))*D2(N)/DPT/DYRAP [GEV**-2]" > "$tmpfile"
    cat "$file" >> "$tmpfile"
    mv "$tmpfile" "$file"
done