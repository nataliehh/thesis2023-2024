# Run as: bash fashion200k_processing.sh ".\data\Fashion200k\women"
root_directory="$1"

# The structure of the fashion data is as follows: 
# The root directory is /women/, from which we get to folders with the main clothing groups (e.g. dresses, pants)
# Within each main clothing group, the items are split into more refined categories (e.g. a dress can be a gown, cocktail dress...)
# Finally, each item has its own folder with an associated item ID, e.g. a specific cocktail dress may have folder 123456
# Then, the folder path to the images of this particular cocktail dress looks something like: /women/dresses/cocktail/123456

# Loop through each main category (e.g. dresses) in the root directory (/women/)
find "$root_directory" -mindepth 2 -maxdepth 2 -type d | while read -r subdirectory; do
    echo "$subdirectory" \;
    # Find the ID folders (e.g. 123456) and move the images from there to the parent folder (e.g. cocktail dress)
    find "$subdirectory" -mindepth 2 -type f -exec mv --backup=numbered {} "$subdirectory" \;
done

# Remove all ID folders (e.g. 123456) that are now empty after the move
find "$root_directory" -mindepth 3 -type d -delete;
