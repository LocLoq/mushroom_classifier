import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Bonus: In ra tên card đồ họa để chắc chắn 100%
if torch.cuda.is_available():
    print(f"Tên Card đồ họa: {torch.cuda.get_device_name(0)}")