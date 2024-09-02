# this script gives back all code that was added to the repo between two points (e.g. SHA or tags) while avoiding useless files
git diff --name-only $1 \
    | egrep "(md|py|yaml|tf|Makefile|Dockerfile|sh|toml|yml|txt|hcl|git)$" \
    | egrep -v "synonymizer/api/modules/" \
    | grep -v "matrix/packages" \
    | while IFS= read -r file; do
    if file -b --mime-type "$file" | grep -q '^text/'; then
        echo -e "\n--- $file ---\n"
        cat "$file"
    fi
done