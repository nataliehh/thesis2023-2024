# The kaggle dataset contains a nested version of itself (a folder that has a copy of itself in it), remove that
rm -r ./data/kaggle_simpsons_characters/simpsons_dataset/simpsons_dataset
# Move all the nested /testset/testset/ data into its parent, /testset/ directory
mv ./data/kaggle_simpsons_characters/kaggle_simpson_testset/kaggle_simpson_testset/* ./data/kaggle_simpsons_characters/kaggle_simpson_testset
# Remove empty nested /testset/testset/ foldery
rm -r ./data/kaggle_simpsons_characters/kaggle_simpson_testset/kaggle_simpson_testset

