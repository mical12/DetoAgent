"""
统计多个 OpenFOAM 案例目录中每个子文件夹（
如 system/、constant/、0/ 等）的文件数量和总行数，并计算多个案例的平均文件数和行数。"""




import os
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def count_files_and_lines(path):
    result = {}
    total_files = 0
    total_lines = 0
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)

        if os.path.isdir(subdir_path) and not (
            (is_number(subdir) and float(subdir) != 0) or 
            subdir.startswith("processor") or 
            subdir.startswith("post") or 
            subdir.startswith("VTK")
        ):
            file_count = 0
            line_count = 0

            for item in os.listdir(subdir_path):
                item_path = os.path.join(subdir_path, item)
                if os.path.isfile(item_path):
                    file_count += 1
                    with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        line_count += len(lines)

            result[subdir] = {'file_count': file_count, 'line_count': line_count}
            total_files += file_count
            total_lines += line_count
    return result, total_files, total_lines

def display_results(results, total_files, total_lines):
    print(f"{'Subdirectory':<40} {'File Count':<15} {'Line Count':<15}")
    print("="*70)
    for subdir, counts in results.items():
        print(f"{subdir:<40} {counts['file_count']:<15} {counts['line_count']:<15}")
    
    print("="*70)
    print(f"{'Total':<40} {total_files:<15} {total_lines:<15}")

ave_files = []
ave_lines = []
n=5
for i in range(n):
    path_dir = f"/data/Chenyx/metaOpenfoam/run/squareBendLiq_{i+1}"
    results, total_files, total_lines = count_files_and_lines(path_dir)
    ave_files.append(total_files)
    ave_lines.append(total_lines)
    display_results(results, total_files, total_lines)

a_file = sum(ave_files)/n
a_lines = sum(ave_lines)/n
print("ave_files:",a_file, "ave_lines",a_lines)