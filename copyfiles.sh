# copy and rename Andreas' files
find . -type f -name 'FluiduMData*.csv' -exec sh -c 'cp "$0" "$(dirname "$0")/fluidumdata.csv"' {} \;
find . -type f -name 'Freeze_out*.csv' -exec sh -c 'cp "$0" "$(dirname "$0")/freezeout.csv"' {} \;