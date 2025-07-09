import os

def count_python_lines(folder):
    total_lines = 0

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        stripped = line.strip()
                        if stripped and not stripped.startswith('#'):
                            total_lines += 1

    return total_lines

folder_path = 'unstable'  # Replace with your folder path
print(f"Total Python lines (excluding comments and empty lines): {count_python_lines(folder_path)}")
