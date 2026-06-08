import urllib.request
import urllib.parse
import time

def test_pollinations(prompt: str, save_as: str = "test_image.png"):
    prompt = prompt[:200]
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed=42"
    
    print(f"Requesting: {url[:80]}...")
    
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # increased to 120s
        status = resp.status
        data = resp.read()
    
    print(f"Status: {status}")
    print(f"Bytes received: {len(data)}")
    
    is_png  = data[:4] == b'\x89PNG'
    is_jpeg = data[:2] == b'\xff\xd8'
    print(f"Valid image: {is_png or is_jpeg} ({'PNG' if is_png else 'JPEG' if is_jpeg else 'UNKNOWN'})")
    
    with open(save_as, "wb") as f:
        f.write(data)
    print(f"Saved to: {save_as}")
    return True


prompts = [
    "Diagram of a multimodal AI platform integrating text, image, and audio processing",
    "Flowchart showing transformer attention mechanism with query key value",
    "System architecture diagram for a RAG pipeline",
]

for i, p in enumerate(prompts, 1):
    print(f"\n--- Test {i} ---")
    try:
        test_pollinations(p, save_as=f"test_{i}.png")
        print("PASS ✅")
    except Exception as e:
        print(f"FAIL ❌ → {e}")
    
    time.sleep(5)  # wait 5s between requests to avoid rate limit