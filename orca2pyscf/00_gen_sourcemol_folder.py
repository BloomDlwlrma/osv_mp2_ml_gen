import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

# Configuration
WORK_DIR = "/home/ubuntu/Shiwei/gen_feature/orca2pyscf"
#SRC_DIR = "/home/ubuntu/Shiwei/qm9_reaction_eng/qm9_xyz_files"
SRC_DIR = "/home/ubuntu/Shiwei/gen_feature/orca2pyscf/xyz_files"
BASIS_SETS = ["cc-pVTZ", "def2-TZVP"]

def clean_xyz(src_path, dest_path):
    with open(src_path, 'r') as f_in:
        lines = f_in.readlines()
    
    if not lines:
        raise ValueError("Empty file")
        
    try:
        natom = int(lines[0].strip())
    except ValueError:
        raise ValueError(f"Invalid atom count in line 1: {lines[0]}")
    
    # Needs: line 0 (count) + line 1 (comment) + natom lines
    lines_needed = natom + 2
    if len(lines) < lines_needed:
        logging.warning(f"File {src_path} has fewer lines ({len(lines)}) than expected ({lines_needed}). Copying all available.")
        raw_lines = lines
    else:
        raw_lines = lines[:lines_needed]

    # Process lines
    out_lines = []
    # 1. Atom count (keep original)
    out_lines.append(raw_lines[0])
    
    # 2. Comment line (clear if it starts with gdb, otherwise keep)
    comment_line = raw_lines[1]
    if comment_line.strip().startswith("gdb"):
        out_lines.append("\n")
    else:
        out_lines.append(comment_line)
    
    # 3. Atom coordinates: keep only first 4 columns (Atom X Y Z)
    for l in raw_lines[2:]:
        parts = l.strip().split()
        if len(parts) >= 4:
            new_line = f"{parts[0]:<5} {parts[1]:>15} {parts[2]:>15} {parts[3]:>15}\n"
            out_lines.append(new_line)
        else:
            out_lines.append(l) # Fallback
        
    with open(dest_path, 'w') as f_out:
        f_out.writelines(out_lines)

def main():
    # Paths
    xyz_dest_dir = os.path.join(WORK_DIR, "xyz_files")
    source_files_dir = os.path.join(WORK_DIR, "source_files")
    
    os.makedirs(xyz_dest_dir, exist_ok=True)
    os.makedirs(source_files_dir, exist_ok=True)
    
    logging.info(f"Target basis sets: {BASIS_SETS}")

    if not os.path.exists(SRC_DIR):
        logging.error(f"Source directory not found: {SRC_DIR}")
        sys.exit(1)

    src_files = [f for f in os.listdir(SRC_DIR) if f.endswith('.xyz')]
    logging.info(f"Found {len(src_files)} XYZ files to process.")
    
    for f in src_files:
        src_f = os.path.join(SRC_DIR, f)
        dest_f = os.path.join(xyz_dest_dir, f)
        mol_name = os.path.splitext(f)[0]
        
        # 1. Clean and Copy to xyz_files
        try:
            clean_xyz(src_f, dest_f)
        except Exception as e:
            logging.error(f"Failed to clean {f}: {e}")
            continue
            
        # 2. Create Directory Structure in source_files
        # Create separate folders for each basis set: mol_name/mol_name_basis
        mol_dir = os.path.join(source_files_dir, mol_name)
        
        for basis in BASIS_SETS:
            folder_name = f"{mol_name}_{basis}"
            basis_dir = os.path.join(mol_dir, folder_name)
            os.makedirs(basis_dir, exist_ok=True)
                
    logging.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
