import time, hmac, hashlib

# Replace with your actual key and secret
API_KEY = "A6ucmdIu9DZCi3ZaDz"
API_SECRET = "M3Zz9RedjrwrC8CF0K8KlHQeHkf3eCpEQMCi"

timestamp = str(int(time.time() * 1000))
params = {
    "timestamp": timestamp,
    "recvWindow": "5000",
    "accountType": "UNIFIED"
}

# Generate query string
query = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
signature = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

print("🔑 API Key:", API_KEY)
print("📅 Timestamp:", timestamp)
print("📦 Query String:", query)
print("🔏 Signature:", signature)

