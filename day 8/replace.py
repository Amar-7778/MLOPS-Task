import os
from pathlib import Path

def process_file(file_path):
    replacements = {
        'person': 'person',
        'human': 'person',
        'people': 'person',
        'box': 'box',
        'container': 'box',
        'ladle': 'ladle',
        'forklift': 'forklift',
    }

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    modified_lines = []
    changed = False

    for line in lines:
        parts = line.split()
        if not parts:
            modified_lines.append(line)
            continue

        class_name = parts[0].strip()
        canonical_name = replacements.get(class_name.lower(), class_name.lower())
        if canonical_name != class_name:
            parts[0] = canonical_name
            changed = True

        modified_lines.append(' '.join(parts) + '\n')

    if changed:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(modified_lines)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                process_file(file_path)

main_folder = Path(r'D:\Academic Projects\Task 3\day 8\labelled data')

def main():
    if main_folder.exists():
        process_folder(str(main_folder))
    print("Processing complete.")


if __name__ == '__main__':
    main()