import math
from collections import Counter

# ====== Đọc dữ liệu từ file Excel ======
import sys
import pandas as pd

TARGET = "Mua máy tính"


def load_data(path="mua_may_tinh.xlsx", sheet_name=0):
    """Đọc file Excel -> (list[dict], danh sách thuộc tính). Cột ID bị bỏ qua."""
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])
    df = df.apply(lambda col: col.str.strip())
    features = [c for c in df.columns if c != TARGET]
    return df.to_dict("records"), features


# ====== Các hàm tính toán ======
def entropy(data, target=TARGET):
    """H(S) = -sum(p_i * log2(p_i))"""
    n = len(data)
    counts = Counter(row[target] for row in data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def information_gain(data, feature, target=TARGET):
    """IG(S, A) = H(S) - sum(|S_v|/|S| * H(S_v))"""
    n = len(data)
    remainder = 0.0
    for value in set(row[feature] for row in data):
        subset = [r for r in data if r[feature] == value]
        remainder += len(subset) / n * entropy(subset, target)
    return entropy(data, target) - remainder


def majority_label(data, target=TARGET):
    return Counter(row[target] for row in data).most_common(1)[0][0]


# ====== Xây dựng cây ID3 ======
def build_tree(data, features, target=TARGET):
    """Trả về cây dưới dạng dictionary: {thuộc_tính: {giá_trị: cây_con | nhãn}}"""
    labels = set(row[target] for row in data)

    # 1. Tất cả mẫu cùng nhãn -> nút lá
    if len(labels) == 1:
        return labels.pop()

    # 2. Hết thuộc tính -> nhãn đa số
    if not features:
        return majority_label(data, target)

    # 3. Chọn thuộc tính có Information Gain lớn nhất
    best = max(features, key=lambda f: information_gain(data, f, target))
    tree = {best: {}}
    remaining = [f for f in features if f != best]

    # 4. Chia nhánh theo từng giá trị của thuộc tính tốt nhất
    for value in set(row[best] for row in data):
        subset = [r for r in data if r[best] == value]
        tree[best][value] = build_tree(subset, remaining, target)

    return tree


# ====== Dự đoán ======
def predict(tree, sample, default=None):
    """Dự đoán nhãn cho một mẫu mới (dict)."""
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = sample.get(feature)
    branches = tree[feature]
    if value not in branches:
        return default  # giá trị chưa gặp khi huấn luyện
    return predict(branches[value], sample, default)


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "->", tree)
        return
    feature = next(iter(tree))
    for value, sub in tree[feature].items():
        print(f"{indent}[{feature} = {value}]")
        print_tree(sub, indent + "    ")


# ====== Chạy thử ======
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "mua_may_tinh.xlsx"
    DATA, FEATURES = load_data(path)
    print(f"Đã đọc {len(DATA)} mẫu từ {path}, thuộc tính: {FEATURES}\n")
    print("Entropy toàn tập: %.4f\n" % entropy(DATA))
    for f in FEATURES:
        print(f"IG({f}) = {information_gain(DATA, f):.4f}")

    tree = build_tree(DATA, FEATURES)
    print("\nCây quyết định (dictionary):")
    print(tree)
    print("\nCây dạng văn bản:")
    print_tree(tree)

    default = majority_label(DATA)
    tests = [
        {"Tuổi": "Trẻ", "Thu nhập": "Thấp", "Sinh viên": "Có", "Đánh giá tín dụng": "Tốt"},
        {"Tuổi": "Trẻ", "Thu nhập": "Cao", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt"},
        {"Tuổi": "Trung niên", "Thu nhập": "Trung bình", "Sinh viên": "Không", "Đánh giá tín dụng": "Xuất sắc"},
        {"Tuổi": "Già", "Thu nhập": "Trung bình", "Sinh viên": "Có", "Đánh giá tín dụng": "Xuất sắc"},
        {"Tuổi": "Già", "Thu nhập": "Thấp", "Sinh viên": "Không", "Đánh giá tín dụng": "Tốt"},
    ]
    print("\nDự đoán mẫu thử:")
    for s in tests:
        print(s, "=>", predict(tree, s, default))

    # Kiểm tra độ chính xác trên tập huấn luyện
    correct = sum(predict(tree, r, default) == r[TARGET] for r in DATA)
    print(f"\nĐộ chính xác trên tập huấn luyện: {correct}/{len(DATA)}")