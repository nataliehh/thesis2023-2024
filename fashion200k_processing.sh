# Run as: bash fashion200k_processing.sh ".\data\Fashion200k\women"
root_directory="$1"

# Loop through each subdirectory in the root directory
find "$root_directory" -mindepth 2 -maxdepth 2 -type d | while read -r subdirectory; do
    # Execute the find and mv command for each subdirectory
    echo "$subdirectory" \;
    find "$subdirectory" -mindepth 2 -type f -exec mv --backup=numbered {} "$subdirectory" \;
done

find "$root_directory" -mindepth 3 -type d -delete;
