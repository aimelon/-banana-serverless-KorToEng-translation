# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests

model_inputs = {'inputs': '13일 인플레이션 공포로 인해 비트코인, 이더리움 등 가상화폐 가격이 폭락세를 보이고 있다.'}

res = requests.post('http://localhost:8000/', json = model_inputs)

print(res.json())