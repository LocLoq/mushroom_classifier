import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# ==========================================
# 1. CẤU HÌNH BAN ĐẦU
# ==========================================
# Khai báo lại danh sách nhãn (phải khớp CHÍNH XÁC thứ tự lúc train)
# Dựa trên log của bạn, thứ tự là:
class_names = ['nam_huong', 'nam_kim_cham'] 
num_classes = len(class_names)

# Đường dẫn tới file mô hình và bức ảnh bạn muốn test
model_path = 'nammushroom_efficientnet_b0.pth'
image_path = 'test2.jpg' # Thay bằng đường dẫn ảnh thật trên máy bạn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. KHÔI PHỤC LẠI MÔ HÌNH (LOAD MODEL)
# ==========================================
print("Đang tải mô hình...")
# Khởi tạo khung xương mô hình rỗng (không cần tải weights từ Google nữa)
model = models.efficientnet_b0(weights=None)

# Chỉnh lại lớp cuối cùng cho khớp với 2 loại nấm của ta
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

# Đắp "não" (trọng số đã học) vào khung xương
# map_location=device giúp code chạy mượt dù bạn đem sang máy không có GPU
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

# Chuyển mô hình sang thiết bị (GPU/CPU) và bật chế độ Đánh giá
model = model.to(device)
model.eval() # QUAN TRỌNG: Tắt các tính năng chỉ dùng khi train (như Dropout)

# ==========================================
# 3. CHUẨN BỊ BỨC ẢNH TEST
# ==========================================
# Bức ảnh test phải trải qua quá trình xử lý y hệt như tập 'val' lúc train
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Mở ảnh bằng thư viện PIL
image = Image.open(image_path).convert('RGB')

# Xử lý ảnh và thêm chiều Batch (vì PyTorch luôn nhận đầu vào dạng [batch_size, channels, height, width])
# unsqueeze(0) sẽ biến ảnh từ [3, 224, 224] thành [1, 3, 224, 224]
input_tensor = transform(image).unsqueeze(0).to(device)

# ==========================================
# 4. CHẠY DỰ ĐOÁN (INFERENCE)
# ==========================================
print("Đang phân tích hình ảnh...")

# Tắt tính toán gradient để tăng tốc độ và tiết kiệm RAM
with torch.no_grad():
    # Đưa ảnh qua mô hình
    outputs = model(input_tensor)
    
    # Tính xác suất (phần trăm tự tin) cho từng class bằng hàm Softmax
    probabilities = F.softmax(outputs[0], dim=0)
    
    # Lấy ra class có xác suất cao nhất
    confidence, predicted_idx = torch.max(probabilities, 0)

# Lấy tên loại nấm từ index
predicted_label = class_names[predicted_idx.item()]
confidence_percent = confidence.item() * 100

# In kết quả ra màn hình
print("-" * 30)
print(f"🍄 Kết quả dự đoán : {predicted_label.upper()}")
print(f"🎯 Độ tự tin        : {confidence_percent:.2f}%")
print("-" * 30)

# In chi tiết xác suất của tất cả các loại
print("Chi tiết xác suất:")
for i, name in enumerate(class_names):
    print(f"- {name}: {probabilities[i].item() * 100:.2f}%")