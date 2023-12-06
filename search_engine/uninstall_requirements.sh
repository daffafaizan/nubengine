#!/bin/bash
#to uninstall run the command below
#bash uninstall_requirements.sh

while read -r line; do
    # Remove comments from the line
    line=$(echo $line | sed 's/#.*//')

    # Remove leading and trailing whitespaces
    line=$(echo $line | sed -e 's/^[ \t]*//' -e 's/[ \t]*$//')

    # Skip empty lines
    [ -z "$line" ] && continue

    # Uninstall the package
    pip uninstall -y $line
done < requirements.txt
