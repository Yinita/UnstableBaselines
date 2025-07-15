import os
from collections import defaultdict

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

def _count_file_lines(path: str) -> int:
    """Return # of non-blank, non-comment lines in a single .py file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(
            1
            for line in f
            if (s := line.strip()) and not s.startswith("#")
        )

def _build_counts(root: str):
    """
    Walk `root` once and return two dicts:
        file_lines[path] -> int
        dir_lines[path]  -> int  (recursive sum of all .py lines under it)
    """
    file_lines: dict[str, int] = {}
    dir_lines: defaultdict[str, int] = defaultdict(int)

    for curr_root, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue

            file_path = os.path.join(curr_root, name)
            n = _count_file_lines(file_path)
            file_lines[file_path] = n

            # bubble the count up every ancestor directory (including root)
            rel_parts = os.path.relpath(file_path, root).split(os.sep)
            for i in range(1, len(rel_parts) + 1):
                dir_path = os.path.join(root, *rel_parts[:i - 1]) if i > 1 else root
                dir_lines[dir_path] += n

    return file_lines, dir_lines

def _print_tree(curr_dir: str, file_lines: dict[str, int],
                dir_lines: dict[str, int], indent: str = "") -> None:
    """Pretty-print one directory and recurse into children."""
    print(f"{indent}{os.path.basename(curr_dir) or curr_dir}/ "
          f"({dir_lines.get(curr_dir, 0)})")

    indent += "    "
    # sort dirs first, then files, each alphabetically
    entries = sorted(os.listdir(curr_dir))
    for entry in entries:
        path = os.path.join(curr_dir, entry)
        if os.path.isdir(path):
            if path in dir_lines:          # only show dirs that contain .py lines
                _print_tree(path, file_lines, dir_lines, indent)
        elif entry.endswith(".py"):
            print(f"{indent}{entry} ({file_lines[path]})")

def count_python_lines_tree(root_folder: str) -> None:
    """
    Build the counts and immediately print the tree view.
    Returns nothing – modify if you need the raw numbers.
    """
    file_lines, dir_lines = _build_counts(root_folder)
    _print_tree(root_folder, file_lines, dir_lines)

# ----------------------------------------------------------------------
# USAGE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    folder_path = "unstable"          # <-- replace or make this a CLI arg
    count_python_lines_tree(folder_path)
    print(f"\n\nTotal line count: {count_python_lines(folder_path)}")

