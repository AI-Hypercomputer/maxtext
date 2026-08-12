import re

with open("/tmp/hlo_dump.hlo") as f:
    text = f.read()

# Search for main/entry function
lines = text.splitlines()
main_lines = []
in_main = False
for l in lines:
    if "func.func public @main" in l or "func.func @main" in l or "ENTRY" in l:
        in_main = True
    if in_main:
        main_lines.append(l)

print(f"Total lines in dump: {len(lines)}")
print(f"Main function lines: {len(main_lines)}")

# Print return statement of main or last lines of main
for l in main_lines[-30:]:
    print("  ", l)
