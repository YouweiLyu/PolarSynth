#!/bin/bash

data_dir="${MITSUBA_DATA_DIR:-./datasets}/test_set/sgl_obj_1213_1k"
targets=("pol000" "pol045" "pol090" "pol135" "normal" "material_tag" "mask")
max_size=1700  # 1.8KB in bytes

for file in $data_dir/pol000/*.png; do
    if [ -f "$file" ]; then
        size=$(stat -c %s "$file")        
        if [ "$size" -lt "$max_size" ]; then
            echo "$file (Size: $size bytes)"
            for target in "${targets[@]}"; do
                newfile=$(echo "$file" | sed "s/pol000/$target/")
                if [ -f $newfile ]; then
                    rm $newfile # dangerous operations
                    echo "Delete $newfile"
                fi
            done
        fi
    fi
done
