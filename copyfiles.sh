find . -type f -name 'FluiduMData*.csv' -exec sh -c 'cp "$0" "$(dirname "$0")/fluidumdata.csv"' {} \;
find . -type f -name 'FreezeOut*.csv' -exec sh -c 'cp "$0" "$(dirname "$0")/freezeout.csv"' {} \;