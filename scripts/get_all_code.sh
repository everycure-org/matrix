git ls-files \
    | egrep "(md|py|yaml|tf|Makefile|Dockerfile|sh|toml|yml|txt|hcl|git)$" \
    | grep -v "matrix/packages" \
    | while IFS= read -r file; do
    if file -b --mime-type "$file" | grep -q '^text/'; then
        echo -e "\n--- $file ---\n"
        cat "$file"
    fi
done