file_names=(
mattdavidow-sweep3-1b-1slice-odd-a4.txt
mattdavidow-sweep3-1p5e-4.txt
mattdavidow-sweep3-1b-2slice-odd-a1.txt
mattdavidow-sweep3-1b-even-a1.txt
mattdavidow-run-names-sweep3-2b.txt)

combined_file="combined_sweep3.txt"
for file in ${file_names[@]}; do
    echo $file
    cat $file >> $combined_file
done

sort -o $combined_file $combined_file
sort $combined_file | uniq > sweep3-run-names.txt
#uniq -u $combined_file > "unique_combined_2.txt"