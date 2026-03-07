#!/bin/bash
set -e

# Instructions:
#
# This script is a Git pre-commit hook that spell checks any content you are about to commit.
#
# If it detects typos, it interactively prompts the user to add the words to the project-local
# wordlist or ignore them.
#
# Adapted from https://github.com/mprpic/git-spell-check for use with the pre-commit framework.
#
# The main difference to the original is that filenames are passed as arguments from pre-commit,
# instead of hardcoding `git diff`.
#
# Each time you try to commit something, this script is run and spell checks the content you are committing.
#
# Should you want to bypass the pre-commit hook (though not recommended), you can commit with "git commit --no-verify".
# Alternatively, type "i" for "ignore" in the interactive session during the commit stage.
#
# Dictionary Configuration:
# - By default, uses project-local dictionary: .aspell-project-dict.txt (created automatically if missing)
# - To use a different project dictionary location, use: --project-dict=path/to/dict.txt (relative to repo root)
# - To also use your personal dictionary, use: --use-global-dict
#   Examples:
#     Command line: /path/to/aspell-check.sh --project-dict=docs/.aspell-dict.txt file1.md file2.md
#     Command line with global: /path/to/aspell-check.sh --use-global-dict file1.md file2.md
#     In .pre-commit-config.yaml:
#       - repo: /home/br/repos/aspell_md
#         hooks:
#           - id: aspell-md
#             files: ^docs/.*\.md$
#             # To use custom project dictionary location:
#             args: ['--project-dict=docs/.aspell-dict.txt']
#             # To include personal dictionary:
#             # args: ['--use-global-dict']
#             # Both:
#             # args: ['--project-dict=docs/.aspell-dict.txt', '--use-global-dict']

# Find git repository root to ensure we use the project-local dictionary
repo_root=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")

# Default values
project_dict_path=".aspell-project-dict.txt"
use_global_dict=false

# User's global dictionary (optional, off by default)
global_dict=~/.git-spell-check

# Language of your doc. When using a non-English language, make sure you have the appropriate aspell libraries
# installed: "yum search aspell". For example, to spell check in Slovak, you must have the aspell-sk package installed.
lang=en

# Define an extension for any additional dictionaries (containing words that are ignored during the spell check) that
# are kept locally in your repository. These dictionaries will be loaded on top of the existing project dictionary.
extension=pws.2

# The following is a temporary dictionary (a binary file) created from the dict text file. It is deleted after the
# script finishes.
temp_dict=$(mktemp /tmp/docs-dictionary-XXXXXX)

# Parse command-line arguments
files=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dict=*)
            project_dict_path="${1#*=}"
            shift
            ;;
        --use-global-dict)
            use_global_dict=true
            shift
            ;;
        *)
            # This is a file to check
            files+=("$1")
            shift
            ;;
    esac
done

# Set full path to project dictionary
project_dict="$repo_root/$project_dict_path"


# Create project dictionary if it doesn't exist
if [ ! -f "$project_dict" ]; then
    # Create parent directory if needed
    mkdir -p "$(dirname "$project_dict")"
    printf "%s\n" "Project dictionary not found. Created $project_dict"
    cat > "$project_dict" << 'EOF'
# Project-specific dictionary for spell checking
# Add one word per line
EOF
fi

# Clean up if script is interrupted or terminated.
trap "cleanup" SIGINT SIGTERM 


# Prepares the dictionary from scratch in case new words were added since last time.
function prepare_dictionary() {

    local_dict=$(find "$repo_root" -name "*.$extension" -exec ls {} \;)

    temp_file=$(mktemp /tmp/temp_file-XXXXXX)

    # Always include project dictionary (filter out empty lines and lines starting with #)
    if [ -f "$project_dict" ]; then
        grep -v '^#' "$project_dict" | grep -v '^[[:space:]]*$' >> "$temp_file" 2>/dev/null || true
    fi

    # Optionally include global dictionary
    if [ "$use_global_dict" = true ] && [ -f "$global_dict" ]; then
        grep -v '^#' "$global_dict" | grep -v '^[[:space:]]*$' >> "$temp_file" 2>/dev/null || true
        printf "%s\n" "Using personal dictionary: $global_dict"
    fi

    # Include any additional .pws.2 dictionaries in the repository
    if [ -n "$local_dict" ]; then
        for file in $local_dict; do
            grep -v '^#' "$file" | grep -v '^[[:space:]]*$' >> "$temp_file" 2>/dev/null || true
        done
    fi

    # Only create aspell dict if temp_file has content
    if [ -s "$temp_file" ]; then
        sort -u "$temp_file" -o "$temp_file"
        aspell --lang="$lang" create master "$temp_dict" < "$temp_file"
    else
        # Create empty aspell dictionary from empty input (creates proper format)
        aspell --lang="$lang" create master "$temp_dict" < "$temp_file"
    fi
    /bin/rm -f "$temp_file"

}


# Removes the temporary dictionary.
function cleanup() {

    /bin/rm -f "$temp_dict"

}


# Spell checks content you're about to commit. Writes out words that are misspelled or exits with 0 (i.e. continues with
# commit).
function spell_check() {

    if [ ${#files[@]} -eq 0 ]; then
        printf "%s\n" "No files to check."
        cleanup; exit 0
    fi
    
    # Collect words from all files using cat + pipe for robustness
    local all_words=""
    for file in "${files[@]}"; do
        if [ -s "$temp_dict" ]; then
            all_words+=$(cat "$file" | aspell --mode=markdown list --lang="$lang" --extra-dicts="$temp_dict")
            all_words+=$'\n'
        else
            all_words+=$(cat "$file" | aspell --mode=markdown list --lang="$lang")
            all_words+=$'\n'
        fi
    done
    
    words=$(echo "$all_words" | sort -u | grep -v '^$')
    
    if [ -z "$words" ]; then
        printf "%s\n" "No typos found. Proceeding with commit..."
        cleanup; exit 0
    fi
    
    printf "%s\n" "Spell check failed on the following words:
-------------------------------------------------"
    echo "$words"
    for word in $words; do
        # For grep, we still use the files directly (pre-commit has already set up the working tree correctly)
        grep --color=always --exclude-dir={.git,tmp} -HIrone "\<$word\>" "${files[@]}" 2>/dev/null | awk -F ":" '{print "File: " $1 "\ton line: " $2 "\tTypo: " $3}' || true
        printf "%s\n" "-------------------"
    done

}


# Adds all, some, or none of the misspelled words to the custom dictionary.
function add_words_to_dict() {

    printf "%s\n" "
Add any of the misspelled words into your project dictionary ($project_dict)?
  * a[ll]     (add all words into dict, continue with commit)
  * s[ome]    (add some words into dict, fix others, no commit)
  * i[gnore]  (add some words into dict, ignore rest, continue with commit)
  * n[one]    (no commit)
"

    while true; do
        exec < /dev/tty # Simply reading user input does not work because Git hooks have stdin detached.
        read answer
        shopt -s nocasematch
        case "$answer" in
            a|all)
                add_all
                cleanup; exit 0
                ;;
            s|some)
                add_some
                printf "%s\n" "Please fix remaining typos, use \"git add\" to add fixed files, and commit."
                cleanup; exit 1
                ;;
            i|gnore)
                add_some
                cleanup; exit 0
                ;;
            n|none)
                add_none
                cleanup; exit 1
                ;;
            *) 
                printf "%s\n" "Incorrect answer. Try again."
                continue
        esac
        shopt -u nocasematch
    done

}


# Helper function to add a word to the dictionary with proper newline handling
function add_word_to_dict() {
    local word="$1"
    # Check if file ends with a newline
    if [ -s "$project_dict" ] && [ "$(tail -c 1 "$project_dict" | wc -l)" -eq 0 ]; then
        # File doesn't end with newline, add one first
        echo "" >> "$project_dict"
    fi
    echo "$word" >> "$project_dict"
}

# Adds all words to the custom dictionary and continues with the commit.
function add_all() {

    for word in $words; do
        add_word_to_dict "$word"
    done
    printf "%s\n" "All words added to $project_dict"
    # Auto-stage the dictionary so changes persist
    git add "$project_dict" 2>/dev/null || true
    printf "%s\n" "Dictionary has been staged for commit."

}


# Adds some (selected by user) of the words to the dictionary and exits with 1.
function add_some() {

    for word in $words; do
        printf "%s\n" "Do you want to add the following word to your project dictionary: $word  (y[es] or n[o])"
        while true; do
            exec < /dev/tty
            read answer
            shopt -s nocasematch
            case "$answer" in
                y|yes)
                    add_word_to_dict "$word"
                    printf "%s\n" "\"$word\" added to $project_dict."
                    break ;;
                n|no)
                    break ;;
                *) 
                    printf "%s\n" "Incorrect answer. Try again."
                    continue
            esac
            shopt -u nocasematch
        done
    done
    # Auto-stage the dictionary so changes persist
    git add "$project_dict" 2>/dev/null || true
    printf "%s\n" "Dictionary has been staged for commit."

}


# Adds none of the words and exits with 1.
function add_none() {

    printf "%s\n" "No words were added to your project dictionary."
    printf "%s\n" "Please fix remaining typos, use \"git add\" to add fixed files, and commit."

}


prepare_dictionary
spell_check
add_words_to_dict
