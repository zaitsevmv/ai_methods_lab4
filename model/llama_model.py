import httpx
import os

from .config_handler import load_config


# Asynchronous function to generate text
async def generate_llama(text: str, config='config_llama.json',
                         temperature=0.9, num_beams=3) -> str:
    data = load_config(config)
    if data is not None:
        temperature = data.get('temperature', temperature)
        num_beams = data.get('num_beams', num_beams)
    
    api_key = os.getenv("LLAMA_API_KEY")  # Get API key from environment variable
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": [{"role": "user", "content": text}],
                "temperature": temperature,
                "num_beams": num_beams,
            }
        )
        # Check if the response status code is 200
        if response.status_code == 200:
            completion = response.json()
            return completion['choices'][0]['message']['content']
        else:
            # Handle the error response
            return "Error occured" 