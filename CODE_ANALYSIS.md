# FAA Code Analysis — Bugs & Architectural Weaknesses

---

## CRITICAL BUGS (crashes or data corruption)

---

### BUG-1 · `transcriber.py` line 114 — Whisper завантажується заново при кожній мові

**Файл**: `backend/transcriber.py`, рядок 114  
**Код**:
```python
def transcribe_segments(audio_path: str) -> list:
    model = _whisper.load_model("base")   # ← ЗАВЖДИ завантажує заново
```
**Проблема**: Функція `transcribe_segments()` ігнорує глобальний кеш `_whisper_model` і завантажує модель щоразу заново. При 5 мовах — 5 повторних завантажень (~150 MB кожного разу). На сервері з 8 GB RAM це може призвести до OOM kill посеред генерації.  
**Виправлення**: Замінити `_whisper.load_model("base")` на `_get_whisper_model()` (функція вже є в цьому ж файлі, рядок 13).

---

### BUG-2 · `pipeline.py` — порожній пул кліпів не виявляється вчасно

**Файл**: `backend/pipeline.py`, рядок 142  
**Код**:
```python
if not os.path.exists(pool_dir) or not os.listdir(pool_dir):
    build_pool(youtube_urls, pool_dir, emit=emit)
```
**Проблема**: Якщо всі 5 YouTube URL не завантажились, `pool_dir` містить лише `clips_index.json`. `os.listdir()` повертає `["clips_index.json"]` — це truthy, тому `build_pool` повторно не викликається. Система витрачає 15-20 хвилин на rewrite + TTS і падає лише під час монтажу з `RuntimeError("No clips prepared")`.  
**Виправлення**: Перевіряти наявність реальних `.mp4` файлів:
```python
existing_clips = [p for p in _glob.glob(os.path.join(pool_dir, "*.mp4"))
                  if not os.path.basename(p).startswith("_src")]
if not existing_clips:
    build_pool(youtube_urls, pool_dir, emit=emit)
```

---

### BUG-3 · `montage.py` — FFmpeg помилки приховані, але `check=True` крашить без контексту

**Файл**: `backend/montage.py`, рядки 116-120, 168-174  
**Код**:
```python
subprocess.run(
    [FFMPEG, ...],
    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600,
)
```
**Проблема**: `stderr=DEVNULL` + `check=True` = якщо FFmpeg падає, виняток `CalledProcessError` не містить жодної інформації про причину. Повідомлення буде просто `"Command 'ffmpeg' returned non-zero exit status 1"`. Налагодити таку помилку неможливо.  
**Виправлення**: Зберігати stderr і показувати його у виключенні:
```python
r = subprocess.run([FFMPEG, ...], capture_output=True, timeout=3600)
if r.returncode != 0:
    raise RuntimeError(f"FFmpeg failed:\n{r.stderr.decode()[-500:]}")
```

---

### BUG-4 · `text_renderer.py` — шрифт може не існувати, але код все одно продовжує

**Файл**: `backend/text_renderer.py`, рядки 14-22  
**Код**:
```python
FONT_PATH = next((f for f in _candidates if os.path.exists(f)), _candidates[-1])
```
**Проблема**: Якщо жоден шрифт не знайдено, `FONT_PATH` стає останнім елементом зі списку незалежно від того, чи він існує. `apply_text_overlays()` потім крашиться з `check=True` — але без корисного повідомлення (stderr DEVNULL). На мінімальному Ubuntu-сервері Liberation fonts можуть бути відсутні.  
**Виправлення**: Додати перевірку:
```python
if not os.path.exists(FONT_PATH):
    raise RuntimeError(f"No font found. Install fonts: apt-get install fonts-liberation")
```

---

### BUG-5 · `clip_matcher.analyze_all_clips` — немає retry при rate limit від Gemini

**Файл**: `backend/clip_matcher.py`, рядки 113-120  
**Код**:
```python
r = client.models.generate_content(model=model, contents=parts)
```
```python
except Exception:
    analysis = {"description": "unknown", "tags": []}
```
**Проблема**: 5 паралельних Gemini-запитів без retry при 429 (rate limit). Кліпи що отримали 429 зберігаються з порожніми тегами (`{"tags": []}`). Такий кліп НІКОЛИ не матчиться жодній секції скрипту і постійно летить у fallback pool незалежно від свого реального вмісту. Якщо 30-40% кліпів отримали 429 при першому аналізі — якість відео значно падає, і помилка непомітна.  
**Виправлення**: Додати retry з backoff:
```python
for attempt in range(3):
    try:
        r = client.models.generate_content(model=model, contents=parts)
        break
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            time.sleep(10 * (attempt + 1))
        else:
            raise
```

---

## ЗНАЧНІ БАГИ (неправильна поведінка, тиха деградація якості)

---

### BUG-6 · `pipeline.py` — clip matching використовує ОРИГІНАЛЬНИЙ транскрипт, не переписаний скрипт

**Файл**: `backend/pipeline.py`, рядок 237  
**Код**:
```python
section_texts_for_match = list(_split_into_chunks(transcript, WORDS_PER_SECTION))
candidates = match_clips_multi(section_texts_for_match, clips_index, ...)
```
**Проблема**: `transcript` — це текст ОРИГІНАЛЬНОГО відео конкурента. `script` (Claude's rewrite) — це те що реально озвучується. Якщо Claude переписав "China built 30 new ports" як "Coastal infrastructure expanded dramatically", то тегове порівняння шукає кліпи для "China built 30 new ports" — але сегменти Whisper містять переписаний текст. Невідповідність між відібраними кліпами і реальним озвученням.  
**Виправлення**: Використовувати `script` (переписаний) для побудови секцій:
```python
section_texts_for_match = list(_split_into_chunks(script, WORDS_PER_SECTION))
```
Але оскільки `candidates.json` кешується між мовами, а script різний для кожної мови — треба кешувати candidates окремо per-language або взагалі не кешувати candidates.

---

### BUG-7 · `rewriter.py` — `MIN_SCRIPT_LENGTH = 20000` спрацьовує майже завжди

**Файл**: `backend/rewriter.py`, рядок 73  
**Код**:
```python
MIN_SCRIPT_LENGTH = 20000
def _expand_script(script, language, video_title):
    if len(script) >= MIN_SCRIPT_LENGTH:
        return script
```
**Проблема**: Для 8-10 хвилинного відео при 145 слів/хвилину потрібно ~1200-1500 слів ≈ 8000-10000 символів. Навіть довгий переписаний скрипт рідко перевищує 15000 символів після одного проходу Claude. Отже `_expand_script` викликається МАЙЖЕ ЗАВЖДИ — додатковий Claude API call при кожному відео. Але гірше: expansion генерується без доступу до оригінального відео — Claude просто продовжує текст "з голови", що може призвести до вигаданих фактів і цифр.  
**Виправлення**: Зменшити поріг до реального розміру потрібного скрипту або взагалі прибрати expansion і натомість просити Claude генерувати достатній об'єм одразу.

---

### BUG-8 · `channel_scanner.py` — `_pick_long_enough` робить окремий API-запит для кожного відео

**Файл**: `backend/channel_scanner.py`, рядок 67  
**Код**:
```python
for v in videos:
    meta = get_video_metadata(v["url"])  # YouTube API call per video
```
**Проблема**: Якщо перші 10 відео занадто короткі, функція робить 10 окремих YouTube Data API запитів. Кожен запит коштує квоти. При 3 API ключах і частих запусках квота може вичерпатись.  
**Виправлення**: Batch-запит через YouTube API (один запит з кількома video IDs замість N окремих).

---

### BUG-9 · `stocks_library._pexels_fallback` — запит включає стоп-слова

**Файл**: `backend/stocks_library.py`, рядок 397  
**Код**:
```python
query = " ".join(section_text.split()[:5])
```
**Проблема**: Перші 5 слів сегменту часто починаються з "The", "In", "As" тощо. Наприклад "The rapid expansion of Chinese" → Pexels шукає "The rapid expansion of Chinese". Результати будуть нерелевантними або порожніми.  
**Виправлення**: Фільтрувати стоп-слова і брати перші 5 значущих слів.

---

### BUG-10 · `transcriber.transcribe_segments` — мердж коротких сегментів без верхньої межі

**Файл**: `backend/transcriber.py`, рядок 121-128  
**Код**:
```python
for seg in raw[1:]:
    dur = seg["end"] - seg["start"]
    if dur < 2.0:
        merged[-1]["end"] = seg["end"]
        merged[-1]["text"] += " " + seg["text"]
    else:
        merged.append(dict(seg))
```
**Проблема**: Якщо підряд іде 10 коротких сегментів (< 2s кожен), всі вони зливаються в один сегмент тривалістю 10-15 секунд. Монтаж тоді ставить один кліп на цей великий сегмент, але кліп зазвичай коротший → виникають gaps і gap-filling loop запускається багато разів.  
**Виправлення**: Додати максимальну тривалість мердженого сегменту (наприклад 5 секунд):
```python
if dur < 2.0 and (merged[-1]["end"] - merged[-1]["start"]) < 5.0:
```

---

### BUG-11 · `pipeline.py` — `_chunk_duration` імпортується але ніде не використовується

**Файл**: `backend/pipeline.py`, рядок 39  
**Код**:
```python
from backend.aligner import _split_into_chunks, _chunk_duration, _get_duration
```
**Проблема**: `_chunk_duration` не використовується ніде в pipeline.py. `_get_duration` також є в `montage.py`. Мертвий імпорт — не баг, але вказує на залишки старого коду.

---

### BUG-12 · `channel_scanner.py` — дата порівнюється за локальним часом сервера, не UTC

**Файл**: `backend/channel_scanner.py`, рядок 130  
**Код**:
```python
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
```
**Проблема**: `datetime.now()` — локальний час сервера (Амстердам, UTC+1/+2). YouTube `upload_date` — UTC. Відео опубліковане о 23:00 UTC може мати `upload_date` сьогодні за UTC, але "вчора" за амстердамським часом. Дрібна помилка що іноді пропускає свіжі відео.  
**Виправлення**: Використовувати `datetime.utcnow()`.

---

### BUG-13 · `app.py` — Flask secret key не змінений у production

**Файл**: `app.py`, рядок 19  
**Код**:
```python
app.secret_key = os.environ.get("FAA_SECRET_KEY", "faa-local-dev-only-change-for-network-use")
```
**Проблема**: Якщо `FAA_SECRET_KEY` не встановлено в `.env`, використовується публічно відомий дефолтний ключ. Будь-хто може підробити Flask сесію.  
**Виправлення**: Перевіряти що ключ встановлений, як це вже зроблено для FAA_USER/FAA_PASS.

---

## MINOR BUGS (рідкісні крайні випадки)

---

### BUG-14 · `tts.py` — polling не обробляє невідомі статуси

TTS API може повернути статуси крім "ending"/"error" (наприклад "processing", "queued"). Поточний код просто чекає до 600 секунд. Якщо API змінить назву статусу — TTS буде висіти 10 хвилин перед краш.

---

### BUG-15 · `stocks_library.pick_stock_clips` — `seen` не оновлюється після Pexels fallback

Pexels кліпи що завантажуються в `general/` не додаються до `seen`. При наступному виклику `pick_stock_clips` для того ж сегменту вони можуть бути повернуті двічі (один раз як Pexels результат, другий раз як локальний кліп).

---

### BUG-16 · `montage._xfade_join` — stdout/stderr не перенаправляються

На відміну від інших FFmpeg викликів, `_xfade_join` не має `stdout=DEVNULL, stderr=DEVNULL`. При великому відео FFmpeg виводить тисячі рядків логів в stderr Flask-процесу.

---

## АРХІТЕКТУРНІ СЛАБКІ МІСЦЯ

---

### ARCH-1 · Послідовне виробництво мов — головний bottleneck часу

**Проблема**: 5 мов виробляються одна за одною. Rewrite (Claude) і TTS — незалежні для кожної мови і могли б виконуватись паралельно. Зараз: 5 × 45 хв = ~3.75 години. З паралельним rewrite+TTS: ~1.5-2 години.  
**Складність виправлення**: Середня. Потребує рефакторингу produce() на async або ThreadPool по мовах (assembly залишити послідовним через CPU).

---

### ARCH-2 · Pexels `general/` папка зростає нескінченно

**Проблема**: Кожен Pexels fallback кліп зберігається назавжди в `stocks/general/`. Після 100 запусків — сотні кліпів на різні теми. `get_all_clips()` повертає всі їх, що робить тегове порівняння шумнішим. Кліп "china factory" і кліп "beach sunset" (завантажений для іншого ніша) мають однакові шанси потрапити в пул.  
**Виправлення**: Або видаляти Pexels кліпи після use, або зберігати їх в niche-specific підпапці.

---

### ARCH-3 · Одне і те саме відео може бути джерелом багаторазово

**Проблема**: Якщо відео конкурента залишається найпопулярнішим кілька днів — система кожного дня вибирає його як джерело. Результат: кілька майже однакових відео на каналі.  
**Виправлення**: Зберігати список вже використаних source URL в niche JSON і виключати їх при наступному скані.

---

### ARCH-4 · Якість тегового матчингу залежить від якості Gemini-аналізу кліпів

**Проблема**: Якщо кліп отримав `{"tags": []}` (через 429 або поганий відеофайл), він ЗАВЖДИ буде в fallback pool і ніколи не матчитиметься. Якщо таких кліпів багато — конкурентна частина відео складається з випадкових кліпів, не пов'язаних з темою.  
**Виправлення**: Відстежувати кількість кліпів з порожніми тегами і попереджати якщо > 20%.

---

### ARCH-5 · `candidates.json` кешується між мовами, але прив'язаний до оригінального транскрипту

**Проблема**: Всі 5 мов використовують один `candidates.json`, збудований на основі ОРИГІНАЛЬНОГО транскрипту (англійського). Для відео польською або турецькою це означає що вибір кліпів заснований на English-language keywords. Теги кліпів також англійські — тому матч ще більш-менш працює. Але якщо Claude кардинально змінює структуру і порядок тем при перекладі — клейки "плавають".

---

### ARCH-6 · Немає валідації якості переписаного скрипту

**Проблема**: Після того як Claude переписує скрипт — немає жодної перевірки: чи він на правильній мові, чи не порожній, чи не містить мета-коментарів ("Here is the rewritten script:"). Якщо Claude повертає щось неправильне — система генерує TTS для сміття і збирає відео.  
**Виправлення**: Мінімальна валідація: довжина > 500 слів, відсутність мета-фраз типу "Here is", "As an AI".

---

### ARCH-7 · `_expand_script` генерує факти без джерела

**Проблема**: Коли скрипт < 20000 символів, Claude генерує continuation без доступу до оригінального відео або будь-яких фактів. Результат: вигадані статистики, неіснуючі цитати, неточні дати — особливо критично для тематики типу "china economy" де цифри важливі.

---

### ARCH-8 · Немає сповіщення при краші в браузер (закрита вкладка = втрачена помилка)

**Проблема**: Помилки відправляються через SocketIO `emit("error", {...})`. Якщо браузер відключився або вкладка закрита — повідомлення про помилку втрачається. Користувач бачить "відео не з'явилось" без розуміння чому.  
**Виправлення**: Зберігати стан і помилку в файл (наприклад `_prepare_{id}/job_status.json`), додати `/api/status` endpoint що повертає останню помилку.

---

## ПІДСУМОК ПРІОРИТЕТІВ

| Пріоритет | Баг/Слабке місце | Вплив |
|---|---|---|
| 🔴 Критично | BUG-1: Whisper завантажується 5 разів | OOM kill сервера |
| 🔴 Критично | BUG-2: Порожній пул не виявляється | 20 хв марної роботи |
| 🔴 Критично | BUG-3: FFmpeg помилки без деталей | Неможливо налагодити |
| 🔴 Критично | BUG-5: Gemini без retry | Тиха деградація якості |
| 🟠 Важливо | BUG-6: candidates на старому транскрипті | Кліпи не збігаються з текстом |
| 🟠 Важливо | BUG-7: expand_script завжди | Зайві API витрати + галюцинації |
| 🟠 Важливо | ARCH-3: Повторне джерело | Дублікати на каналі |
| 🟡 Середньо | BUG-9: Стоп-слова в Pexels | Погані результати пошуку |
| 🟡 Середньо | ARCH-2: Pexels general росте | Шум у матчингу |
| 🟡 Середньо | ARCH-6: Немає валідації скрипту | Сміттєве відео без попередження |
