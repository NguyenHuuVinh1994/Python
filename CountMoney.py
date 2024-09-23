x = int(input('Nhập vào số tiền của bạn: '))
print(f'Số tiền của bạn là: {x}')
Total = 0
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
TongSoTo = Total + To500 + To200 + To100 + To50 + To20 + To10 + To5 + To2 + To1
print(f'Loại 500 gồm {To500} tờ')
print(f'Loại 200 gồm {To200} tờ')
print(f'Loại 100 gồm {To100} tờ')
print(f'Loại 50 gồm {To50} tờ')
print(f'Loại 20 gồm {To20} tờ')
print(f'Loại 10 gồm {To10} tờ')
print(f'Loại 5 gồm {To5} tờ')
print(f'Loại 2 gồm {To2} tờ')
print(f'Loại 1 gồm {To1} tờ')
print(f'Tổng cộng có {TongSoTo} tờ')
input("Press Enter to exit...")
