# Gemini Web Image Bridge

This is an optional local bridge for generating thumbnail images through the
Gemini Web session. It binds to `127.0.0.1` and requires a separate local API
key. Google cookies stay only in `gemini_bridge/.env`.

## Setup

1. Copy `.env.example` to `.env`.
2. Set a long random `LOCAL_API_KEY`.
3. Fill `GEMINI_1PSID` and `GEMINI_1PSIDTS` locally from the signed-in Gemini
   browser session. Do not put these values into FAA settings or commit `.env`.
4. Put the same `LOCAL_API_KEY` into FAA Settings under **Gemini Web Image
   Bridge**, enable image generation, and save.

Start the bridge and FAA together from the repository root with:

```powershell
.\start_faa_with_gemini.ps1
```

The bridge exposes only the local `/health`, `/v1/models`, and
`/v1/images/generations` endpoints. Generated images are stored in each FAA
project as `thumbnail_generated.png` and are included by the downloader as
`thumbnail.png`.

