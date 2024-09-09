#!/bin/bash

rm -rf ./release_web
mkdir ./release_web

rm -rf ./tar_tmp
mkdir -p ./tar_tmp/tutorial
cp -r Tutorial*/ ./tar_tmp/tutorial
cp -r figures/ ./tar_tmp/tutorial
cd ./tar_tmp
tar -zcf tutorial.tar.gz tutorial/
cd ..

mv ./tar_tmp/tutorial.tar.gz ./release_web
rm -rf ./tar_tmp

cp -r ./figures ./release_web
cp -r Tutorial*/ ./release_web

process_directory() {
    local dir="$1"
    echo "Entering directory: $dir"
    find "$dir" -maxdepth 1 -type f -name "*.ipynb" | while read -r file; do
        mkdir -p "./release_web/${dir}"
        jupyter nbconvert --to html "$file"
        html_file="${file%.ipynb}.html"
        cp "${file}" "./release_web/${dir}"
        mv "${html_file}" "./release_web/${dir}"
    done
}

export -f process_directory

find . -maxdepth 1 -type d -name "Tutorial*" | parallel process_directory
python release_md.py
