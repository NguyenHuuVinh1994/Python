a = int(input('Số tiền cần trả:'))
b = int(input('Số tiền khách trả:'))
x = b - a
To500 = x//500000
x=x%500000
To200 = x//200000
x=x%200000
To100 = x//100000
x=x%100000
To50 = x//50000
x=x%50000
To20 = x//20000
x=x%20000
To10 = x//10000
x=x%10000
To5 = x//5000
x=x%5000
To2 = x//2000
x=x%2000
To1 = x//1000
x=x%1000
SoTienThoi = x + To500 + To200 + To100 + To50 + To20 + To10 + To5 + To2 + To1
if a > b:
    print(f'Số tiền khách còn thiếu là {a - b}')
if a == b:
    print('Cám ơn khách hàng. Hẹn gặp lại')
if b > a:
    print(f'Số tiền cần thối là {SoTienThoi}')
    if SoTienThoi == 0:
        print('Không tìm được mệnh giá bạn đưa')
    if To500 > 0:
        print(f'Loại 500 gồm {To500} tờ')
    if To200 > 0:
        print(f'Loại 200 gồm {To200} tờ')
    if To100 > 0:
        print(f'Loại 100 gồm {To100} tờ')
    if To50 > 0:
        print(f'Loại 50 gồm {To50} tờ')
    if To20 > 0:
        print(f'Loại 20 gồm {To20} tờ')
    if To10 > 0:
        print(f'Loại 10 gồm {To10} tờ')
    if To5 > 0:
        print(f'Loại 5 gồm {To5} tờ')
    if To2 > 0:
        print(f'Loại 2 gồm {To2} tờ')
    if To1 > 0:
        print(f'Loại 1 gồm {To1} tờ')
    if SoTienThoi > 0:
        print(f'Tổng cộng có {SoTienThoi} tờ')
print("Cám ơn khách hàng. Hẹn gặp lại")
input("Press Enter to exit...")