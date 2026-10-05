import numpy as np


class Perceptron:
    """Perceptron Learning Algorithm (PLA) cho phân lớp nhị phân, nhãn ∈ {+1, -1}."""

    def __init__(self, max_epochs=1000, random_state=None):
        self.max_epochs = max_epochs      # số lượt duyệt tối đa qua dữ liệu
        self.random_state = random_state  # để tái lập kết quả khi xáo trộn
        self.w = None                     # vector trọng số, phần tử cuối là bias
        self.n_updates = 0                # tổng số lần cập nhật w
        self.converged = False            # True nếu không còn điểm nào sai

    @staticmethod
    def _add_bias(X):
        """Thêm cột 1 vào cuối X: [x1, x2] -> [x1, x2, 1]."""
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        Xb = self._add_bias(X)
        n_samples, n_features = Xb.shape

        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(n_features)
        self.n_updates = 0
        self.converged = False

        for epoch in range(self.max_epochs):
            errors = 0
            for i in rng.permutation(n_samples):       # duyệt theo thứ tự ngẫu nhiên
                if y[i] * (self.w @ Xb[i]) <= 0:       # điểm bị phân lớp sai
                    self.w += y[i] * Xb[i]             # w <- w + y*x
                    self.n_updates += 1
                    errors += 1
            if errors == 0:                            # đã phân lớp đúng toàn bộ
                self.converged = True
                break
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = self._add_bias(X) @ self.w
        return np.where(scores >= 0, 1, -1)


if __name__ == "__main__":
    # Dữ liệu khả tách tuyến tính: hai đám điểm
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=[4, 4], scale=0.8, size=(50, 2))
    X_neg = rng.normal(loc=[0, 0], scale=0.8, size=(50, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [-1] * 50)

    model = Perceptron(max_epochs=100, random_state=42)
    model.fit(X, y)

    print("w (gồm bias ở cuối):", model.w)
    print("Hội tụ:", model.converged, "| số lần cập nhật:", model.n_updates)
    print("Độ chính xác:", np.mean(model.predict(X) == y))
    print("Dự đoán điểm mới (3,3) và (1,0):", model.predict([[3, 3], [1, 0]]))