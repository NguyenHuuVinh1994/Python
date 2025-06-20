import os

# Đường dẫn đến thư mục chứa các file
input = "D:\\Photo"
output = "D:\\Photo1"
# Lặp qua tất cả các file trong thư mục
for filename in os.listdir(input):
    # Kiểm tra nếu file có phần mở rộng hợp lệ
    if filename.endswith(('.jpg', '.jpeg', '.png', '.tif')):
        # Tạo tên file mới bằng cách thay thế "WEB_" và "_RGB"
        new_name = filename.replace("_RGB", "")
        
        # Đường dẫn đầy đủ của file cũ và file mới
        old_file = os.path.join(input, filename)
        new_file = os.path.join(output, new_name)
        
        # Kiểm tra nếu file mới đã tồn tại
        if not os.path.exists(new_file):
            # Đổi tên file
            os.rename(old_file, new_file)
        else:
            print(f"File {new_file} đã tồn tại, bỏ qua đổi tên {old_file}")

print("Đã đổi tên file thành công!")