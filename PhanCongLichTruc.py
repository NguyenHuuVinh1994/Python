import pandas as pd
import pulp
import os
file_path = r'Address'
try:
    schedule_df = pd.read_excel("NameFile.xlsx", sheet_name="Choice")
    print("File loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")

nhan_vien = list(schedule_df.columns[1:])
days_in_month = pd.to_datetime(schedule_df['Ten nguoi truc'], format='%m/%d/%Y')
problem = pulp.LpProblem("Phan_Cong_Lich_Truc", pulp.LpMinimize)
# Tao bien quyet dinh cho moi ngay va moi nhan vien
x = pulp.LpVariable.dicts("x", [(day, nv) for day in days_in_month for nv in nhan_vien], 0, 1, pulp.LpBinary)
# Moi ngay co 1 nhan vien truc.
for day in days_in_month:
    problem += pulp.lpSum(x[(day, nv)] for nv in nhan_vien) == 1
# Neu nhan vien co dang ky nghi phep truoc, thi khong xep lich vao ngay do. 
for day in days_in_month:
    for nv in nhan_vien:
        if schedule_df.loc[schedule_df['Ten nguoi truc'] == day.strftime('%m/%d/%Y'), nv].values[0] == 'Nghi phep':
            problem += x[(day, nv)] == 0
# Nhan vien sau khi truc xong thi n ngay sau do moi truc lai
n = 5
for i in range(len(days_in_month) - n):
    for nv in nhan_vien:
        for j in range(1, n + 1):
            problem += x[(days_in_month[i], nv)] + x[(days_in_month[i + j], nv)] <= 1
# Neu nhan vien duoc xep truc thu 7 hoac chu nhat tuan nay, thi se khong xep truc vao thu 7 hoac chu nhat tuan sau.
for i, day in enumerate(days_in_month):
    if day.weekday() == 5 or day.weekday() == 6:
        saturday_next_week, sunday_next_week = None, None
        for j in range(i + 1, len(days_in_month)):
            if days_in_month[j].weekday() == 5:
                saturday_next_week = days_in_month[j]
                break
        for j in range(i + 2, len(days_in_month)):
            if days_in_month[j].weekday() == 6: 
                sunday_next_week = days_in_month[j]
                break
        for nv in nhan_vien:
            if saturday_next_week is not None:
                problem += x[(day, nv)] + x[(saturday_next_week, nv)] <= 1
            if sunday_next_week is not None:
                problem += x[(day, nv)] + x[(sunday_next_week, nv)] <= 1              
# Rang buoc: Can bang so luong ca truc giua cac nhan vien
total_days = len(days_in_month)
min_shifts = total_days // len(nhan_vien)
max_shifts = min_shifts + 1
for nv in nhan_vien:
    problem += pulp.lpSum(x[(day, nv)] for day in days_in_month) >= min_shifts
    problem += pulp.lpSum(x[(day, nv)] for day in days_in_month) <= max_shifts
problem.solve()
