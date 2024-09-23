Name = str(input('Nhập tên của bạn:'))
Age = int(input('Nhập tuổi của bạn:'))
import datetime
yearnow = datetime.datetime.now().year
a = yearnow + 100 - Age
print(f'Đến năm {a}, bạn {Name} sẽ tròn 100 tuổi')
input('Press Enter to Exit...')