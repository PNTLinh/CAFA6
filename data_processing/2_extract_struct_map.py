import gzip
import numpy as np
from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from scipy.spatial import distance_matrix

def get_contact_map_from_gz(pdb_gz_path, distance_threshold=8.0):
    """
    Đọc file .pdb.gz, trích xuất tọa độ C-alpha và trả về Ma trận kề (Adjacency Matrix).
    """
    parser = PDBParser(QUIET=True) # QUIET=True để tắt các cảnh báo lặt vặt của PDB
    
    # 1. Đọc trực tiếp từ file .gz không cần giải nén ra ổ cứng
    with gzip.open(pdb_gz_path, 'rt') as f:
        structure = parser.get_structure('protein', f)
        
    # 2. Rút trích tọa độ (X, Y, Z) của các nguyên tử Carbon-Alpha (CA)
    ca_coords = []
    
    # Lặp qua các mô hình, chuỗi và gốc axit amin
    for model in structure:
        for chain in model:
            for residue in chain:
                # Bỏ qua các phân tử nước (HOH) hoặc các gốc dị thể, chỉ lấy Axit Amin chuẩn
                if residue.has_id('CA') and residue.id[0] == ' ':
                    coord = residue['CA'].get_coord()
                    ca_coords.append(coord)
                    
    ca_coords = np.array(ca_coords)
    
    # 3. Tính Bản đồ khoảng cách (Distance Matrix)
    # distance_matrix tính khoảng cách Euclide giữa mọi cặp điểm với nhau
    dist_matrix = distance_matrix(ca_coords, ca_coords)
    
    # 4. Chuyển thành Bản đồ tiếp xúc (Contact Map / Adjacency Matrix)
    # Nếu khoảng cách < 8.0 Angstrom thì cho giá trị 1 (có cạnh), ngược lại là 0
    contact_map = (dist_matrix < distance_threshold).astype(int)
    
    # Xóa các cạnh tự nối chính nó (đường chéo chính = 0)
    np.fill_diagonal(contact_map, 0)
    
    return ca_coords, dist_matrix, contact_map

# === TEST THỬ TRÊN 1 FILE ===
if __name__ == "__main__":
    # Thay đường dẫn này bằng 1 file .gz bất kỳ trong máy bạn
    sample_file = "D:/CAFA6/raw_data/struct_feature/AF-A0A0A0MS03-F1-model_v6.pdb.gz"
    
    if Path(sample_file).exists():
        coords, d_map, c_map = get_contact_map_from_gz(sample_file)
        print(f"Protein có {len(coords)} axit amin (Nút).")
        print(f"Kích thước Ma trận kề (Edges): {c_map.shape}")
        print("\nMột góc của Ma trận kề (Contact Map):")
        print(c_map[:5, :5])
    else:
        print("Vui lòng kiểm tra lại đường dẫn file mẫu!")