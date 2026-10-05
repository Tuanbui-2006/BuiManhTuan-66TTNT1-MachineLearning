import numpy as np

# Dữ liệu
w = np.array([-2, 1, 0], dtype=float)
x = np.array([2, 3, 1], dtype=float)   # đã thêm bias
y = 1
eta = 1                                # tốc độ học

# 1. Kiểm tra phân lớp
score = w @ x
pred = 1 if score >= 0 else -1
print(f"w^T x = {score}")
print(f"Dự đoán = {pred}, nhãn thật y = {y}")
for i in range(100):
    if y * score <= 0:
        print("=> Mẫu bị phân lớp SAI")

        # 2. Cập nhật Perceptron
        w_new = w + eta * y * x
        print(f"w mới = {w_new}")

        # 3. Tính lại w^T x
        score_new = w_new @ x
        print(f"w_new^T x = {score_new}")
        print("=> Phân lớp đúng" if y * score_new > 0 else "=> Vẫn sai")
        w = w_new
        score = score_new
    else:
        print("=> Mẫu được phân lớp ĐÚNG, không cần cập nhật")
        print("Số lần lặp cập nhật:", i)
        break