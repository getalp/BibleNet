lang_array_a=("english") # Source language
lang_array_b=("basque" "english" "finnish" "french" "hungarian" "romanian" "russian" "spanish") # Target language

for lang_a in "${lang_array_a[@]}"
do
    echo -en $lang_a"\t"
    for lang_b in "${lang_array_b[@]}"
    do
        #if [ "$lang_a" != "$lang_b" ]; then
        python3 build_splits.py ./data/recap_verses-anName-True.csv $lang_a $lang_b True True 0.8 0.1 0.1 325
        #fi
    done
    echo
done
