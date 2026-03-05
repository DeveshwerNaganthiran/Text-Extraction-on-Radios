import requests
import os

API_KEY = os.getenv("MSI_API_KEY", "GTy:YsSiQSt,cxCGOLsj(ZkjCZDFTh!OkML9WrEn")

url = "https://genai-service.stage.commandcentral.com/app-gateway/api/v2/chat"
headers = {
	"x-msi-genai-api-key": API_KEY,
    "Content-Type": "application/json"
}
json_data = {
    "userId": "bgvk38@motorolasolutions.com",
    "model": "Claude-Sonnet-4",
    "prompt": "Best CS2 competitive team currently",
    "system": "",
	"modelConfig": {
			"temperature": 0.5,
			"max_tokens": 800,
			"top_p": 1,
			"frequency_penalty": 0,
			"presence_penalty": 0    
	}

}

response = requests.post(url, headers=headers, json=json_data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())