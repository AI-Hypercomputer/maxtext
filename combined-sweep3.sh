file_names=(
mattdavidow-sweep3p1-a2.txt
mattdavidow-sweep3p1-baseline-a2.txt)

combined_file="sweep3p1_combined_1.txt"
for file in ${file_names[@]}; do
    echo $file
    cat $file >> $combined_file
done

sort -o $combined_file $combined_file
sort $combined_file | uniq > sweep3p1-run-names.txt
#uniq -u $combined_file > "unique_combined_2.txt"