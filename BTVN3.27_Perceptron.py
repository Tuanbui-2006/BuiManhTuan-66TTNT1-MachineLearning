import numpy as np

# Dữ liệu đề bài
w = np.array([1, 2, -10])   # vector trọng số (phần tử cuối là bias)
x = np.array([3, 4, 1])     # vector đặc trưng (phần tử cuối = 1 là bias đã thêm)
y_true = -1                 # nhãn thực tế

# 1. Tính w^T x
wx = np.dot(w, x)
print("1. w^T x =", wx)

# 2. Xác định nhãn dự đoán (hàm sign)
def sign(z):
    return 1 if z >= 0 else -1

y_pred = sign(wx)
print("2. Nhãn dự đoán y_pred =", y_pred)

# 3. Kiểm tra có bị phân lớp sai không
is_misclassified = (y_pred != y_true)
print("3. Có bị phân lớp sai không?", "CÓ" if is_misclassified else "Không")