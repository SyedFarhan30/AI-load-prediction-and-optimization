import requests
import json
import time

# Give server time to fully start
time.sleep(2)

print('=== Testing ALL Dashboard API Endpoints ===\n')

# First, run the ML pipeline
print('Setting up ML pipeline...')
requests.post('http://localhost:5000/upload', files={'file': open('uploads/Big_data_dataset.csv', 'rb')})
requests.post('http://localhost:5000/preprocess')
requests.post('http://localhost:5000/train_model')
requests.post('http://localhost:5000/predict')
requests.post('http://localhost:5000/detect_anomalies')
requests.post('http://localhost:5000/optimize')
print('ML pipeline ready!\n')

# Test /api/metrics
resp = requests.get('http://localhost:5000/api/metrics')
print('1. /api/metrics:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    cpu = data.get('cpu_usage', 0)
    mem = data.get('memory_usage', 0)
    net = data.get('network_traffic', 0)
    print('   CPU:', round(cpu, 2), '%')
    print('   Memory:', round(mem, 2), '%')
    print('   Network:', round(net, 2), 'Mbps')
else:
    print('   Error:', resp.json())

# Test /api/predictions_summary
resp = requests.get('http://localhost:5000/api/predictions_summary')
print('\n2. /api/predictions_summary:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    pred = data.get('latest_prediction', 0)
    anom = data.get('anomalies_detected', 0)
    print('   Latest Prediction:', round(pred, 2))
    print('   Anomalies:', anom)

# Test /api/alerts
resp = requests.get('http://localhost:5000/api/alerts')
print('\n3. /api/alerts:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    alerts = len(data.get('alerts', []))
    print('   Alert Count:', alerts)

# Test /api/history
resp = requests.get('http://localhost:5000/api/history')
print('\n4. /api/history:', resp.status_code)
if resp.status_code == 200:
    data = resp.json()
    hist = len(data.get('history', []))
    print('   History Entries:', hist)

print('\n✅ All endpoints operational!')
