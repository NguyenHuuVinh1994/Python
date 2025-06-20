import pandas as pd
import requests
import json
from datetime import datetime
import unicodedata

def remove_surrogates(input_string):
    return ''.join(c for c in input_string if not unicodedata.category(c).startswith('Cs'))

# Requests API từ website
apiKey = '0c9ca408712f22c979dea5230a2aa15a'
url = f'https://data.fixer.io/api/latest?access_key={apiKey}&symbols=USD,VND,SGD'

# Vòng lặp kiểm tra phản hồi từ websites
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    if 'rates' in data:
        usd_to_vnd = data['rates'].get('VND', None)
        usd_to_sgd = data['rates'].get('SGD', None)
        
        if usd_to_sgd and usd_to_vnd:
            exchange_rates = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'USD_To_VND': usd_to_vnd,
                'USD_To_SGD': usd_to_sgd
            }
            
            # Xử lý các ký tự không hợp lệ trước khi ghi
            cleaned_exchange_rates = {key: remove_surrogates(str(value)) for key, value in exchange_rates.items()}
            
            # Chuyển đổi dữ liệu thành DataFrame
            df = pd.DataFrame([cleaned_exchange_rates])
            
            # Lưu DataFrame vào file Excel
            df.to_excel('Address', index=False)
                
            print("Update success, and file saved to Excel")
        else:
            print("Exchange rate not found")
    else:
        print("No response from API")
except requests.exceptions.RequestException as e:
    print(f'Error request API: {e}')
except Exception as e:
    print(f'Error: {e}')
