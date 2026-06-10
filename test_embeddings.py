"""Quick test: does Pioneer support /v1/embeddings?"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import config

settings = config.load_settings()
api_keys = settings.get("pioneer_api_keys", [])
if isinstance(api_keys, str):
    api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]

chat_url = settings.get("pioneer_api_url", "")
embed_url = chat_url.replace("chat/completions", "embeddings")

print(f"Chat URL:  {chat_url}")
print(f"Embed URL: {embed_url}")
print(f"Keys: {len(api_keys)} configured")
print()

if not api_keys:
    print("ERROR: no pioneer_api_keys in settings")
    sys.exit(1)

key = api_keys[0]
payload = json.dumps({
    "model": "text-embedding-004",
    "input": ["hello world test"],
}).encode("utf-8")

print("Testing Pioneer /v1/embeddings...")
try:
    req = urllib.request.Request(
        embed_url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    data = body.get("data", [])
    if data and data[0].get("embedding"):
        vec = data[0]["embedding"]
        print(f"SUCCESS! Pioneer supports embeddings.")
        print(f"Vector length: {len(vec)} dimensions")
        print(f"First 5 values: {vec[:5]}")
    else:
        print(f"FAIL: got response but no embedding data: {json.dumps(body)[:200]}")
except Exception as e:
    print(f"FAIL: Pioneer doesn't support embeddings: {e}")
    print()
    print("Trying Vertex AI fallback...")
    try:
        from google import genai
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", config.VERTEX_CREDENTIALS)
        client = genai.Client(
            vertexai=True,
            project=settings.get("vertex_project_id", ""),
            location=settings.get("vertex_location", "us-central1"),
        )
        resp = client.models.embed_content(model="text-embedding-004", contents=["hello world test"])
        embs = getattr(resp, "embeddings", [])
        if embs:
            vec = list(getattr(embs[0], "values", embs[0]))
            print(f"SUCCESS! Vertex works as fallback.")
            print(f"Vector length: {len(vec)} dimensions")
        else:
            print("FAIL: Vertex returned empty")
    except Exception as e2:
        print(f"FAIL: Vertex also broken: {e2}")
