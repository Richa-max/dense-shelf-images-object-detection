import requests, json, base64, os, sys

url = 'http://127.0.0.1:5000/classify_crop'
img_path = os.path.join(os.path.dirname(__file__), 'sample_crop.jpg')
if not os.path.exists(img_path):
    print('ERROR: sample image not found at', img_path)
    sys.exit(1)

files = {'image': open(img_path, 'rb')}
data = {'use_clip': '1'}

r = requests.post(url, files=files, data=data, timeout=300)
print('HTTP', r.status_code)
try:
    j = r.json()
    print(json.dumps(j, indent=2))
except Exception:
    print(r.text)
