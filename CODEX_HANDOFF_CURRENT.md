# FAA: CURRENT CODEX HANDOFF

Останнє оновлення: 2026-08-10.

Це головний handoff для наступного Codex. Прочитати його повністю перед
змінами. Старий HANDOFF.md може бути застарілим; джерело правди -- цей файл
і фактичний код у репозиторії.

## 1. Проєкт

FAA -- Flask-софт для виробництва YouTube-відео про російсько-українську війну
різними мовами.

Flow:
1. YouTube URL -> transcript/subtitles.
2. Rewrite/translation через OpenAI-compatible API.
3. Script, metadata і thumbnail prompt для кожної мови.
4. VoiceGen TTS.
5. Whisper/таймінги аудіо.
6. Вибір кліпів з war library через embeddings/cosine search.
7. FFmpeg montage/final MP4.
8. Ready project на сайті.
9. Windows downloader забирає MP4, metadata, project.json і thumbnail.

Основний niche: russia_ukraine_war.
Типові мови: pl, tr, cs, ro, hu, sv, fi, hr, da, bg.

## 2. Шляхи

Головний локальний код:

~~~text
C:\Users\Ukraine\FAA
~~~

Старий дублікат збережено тут:

~~~text
C:\Users\Ukraine\колишній софт
~~~

Працювати треба в FAA, не в історичній FAA_Linux.

Сервер:

~~~text
/workspace/FAA
~~~

G:\My Drive\workspace і G:\My Drive\FAA -- сховище media/data або backup,
не гарантовано повна актуальна копія коду. Код FAA -- в папці FAA.

Windows defaults у config.py очікують:

~~~text
G:\My Drive\FAA\stocks
G:\My Drive\FAA\movies
~~~

На ПК друга ці шляхи можуть бути відсутні; змінити їх через Settings на
реальні локальні paths. H: має найбільше місця, але близько 133 GB вільного,
тому не копіювати всю library без перевірки.

## 3. Git і робоче дерево

HEAD на момент handoff:

~~~text
2fffa7c Allow parallel downloaders with separate state files
~~~

Worktree не чисте. Є незакомічені зміни/нові файли, зокрема:

~~~text
app.py
backend/war_pipeline.py
backend/gemini_image.py
config.py
download_ready_from_site.py
run.py
start_faa_auto_download.ps1
start_faa_with_gemini.ps1
templates/index.html
templates/settings.html
gemini_bridge/
thumbnail_tests/
~~~

Є також tmp_*.py, diag_*.py, metadata_bundle.py, war_analyze_clips.py та інші
untracked helpers. Не видаляти їх автоматично.

Перевірка:

~~~powershell
cd C:\Users\Ukraine\FAA
git status --short
~~~

Правила:
- Не робити git add .
- Додавати лише конкретні файли.
- Не робити git reset --hard або git checkout --.
- Не відкидати чужі зміни.
- Не комітити data/settings.json: там API/TTS secrets.
- Не комітити .env, gemini_bridge/.env, cookies або raw responses.
- Якщо git pull на сервері блокується untracked files, спершу backup/mv, не
  видаляти навмання.

Локальні Gemini-зміни та частина міграції можуть ще не бути в GitHub. Перед
clone/pull перевірити наявність bridge-файлів і не вважати GitHub повною копією
без перевірки.

## 4. Secrets і settings

Секрети локально:

~~~text
C:\Users\Ukraine\FAA\data\settings.json
C:\Users\Ukraine\FAA\.env
C:\Users\Ukraine\FAA\gemini_bridge\.env
~~~

Не показувати значення в чаті, не вставляти у handoff і не комітити. На новому
ПК ключі вводити вручну. data/settings.json є runtime-конфігом; config.py
містить defaults, але реальні збережені settings можуть відрізнятися.

## 5. Основні параметри

Відео за defaults:

~~~text
1920x1080, 30 fps
~~~

Реальний MP4 перевіряти ffprobe, не розміром файла. Downloader використовує
.part і валідацію, бо маленький MP4 може бути пошкодженим/недокачаним.

Rewrite:
- active provider default: a6api;
- URL: https://a6api.com/v1/chat/completions;
- model default: gpt-5.5;
- A6API reasoning_effort: high;
- max_tokens: 12000;
- alternatives: byesu і custom;
- rewrite_chunks default: 6.

Byesu раніше повертав 403 insufficient_user_quota. A6API давав timeout 524
і інколи відповідь без content. Це API/provider проблеми. Не міняти модель
посеред production і не чекати, що старий текст автоматично перепишеться.

Chunking потрібен для великих transcript. Стискання має бути зв’язним, не
механічним видаленням кількох речень. Не міняти цю логіку без малого тесту.

TTS:
~~~text
URL: https://qw1voicegencore.pro
engine: elevenLabsV3
~~~

Ключі й voice profiles у settings. На сервері profiles можна налаштувати:

~~~bash
cd /workspace/FAA
python3 set_voice_profiles.py
~~~

На чистому ПК voice IDs не з’являться без settings; перевірити UI.

Metadata/title/description/tags мають бути мовою відео. На новому ПК спочатку
зробити один тест і перевірити metadata.txt та project.json.

## 6. Gemini Web Image Bridge

Необов’язковий локальний bridge для thumbnail через залогінений Gemini Web.
Це не офіційний Gemini API, а нестабільний web protocol. Cookies потрібні
тільки bridge.

Файли:

~~~text
backend/gemini_image.py
gemini_bridge/server.py
gemini_bridge/.env.example
gemini_bridge/README.md
gemini_bridge/start_bridge.ps1
start_faa_with_gemini.ps1
~~~

Створити локально:

~~~text
C:\Users\Ukraine\FAA\gemini_bridge\.env
~~~

Вміст за прикладом:

~~~dotenv
HOST=127.0.0.1
PORT=4981
LOCAL_API_KEY=довгий_локальний_ключ
GEMINI_1PSID=значення_з_Gemini_web_session
GEMINI_1PSIDTS=значення_з_Gemini_web_session
GEMINI_MODEL=gemini-3.1-flash-image
GEMINI_LANGUAGE=en
REQUEST_TIMEOUT_SECONDS=360
MAX_IMAGE_BYTES=52428800
~~~

У FAA Settings вказати той самий LOCAL_API_KEY, URL
http://127.0.0.1:4981, увімкнути image generation і вибрати
gemini-3.1-flash-image. Google cookies у FAA Settings не вставляти.

Поточна web-сесія бачила моделі:

~~~text
gemini-2.5-flash-image
gemini-2.5-flash-image-preview
gemini-3-pro-image
gemini-3-pro-image-preview-11-2025
gemini-3.1-flash-image
gemini-3.1-flash-image-preview
~~~

Успішно працювала gemini-3.1-flash-image. Старі gemini-3-pro-image та
gemini-3-pro-image-preview-11-2025 повертали 502 без generated image media.
Причина може бути cookies, web session, модель або зміна протоколу.

Один успішний тест повернув image/jpeg при імені .png. backend/gemini_image.py
нормалізує JPEG/WebP/PNG через Pillow і зберігає реальний PNG як
thumbnail_generated.png.

Bridge:

~~~powershell
cd C:\Users\Ukraine\FAA\gemini_bridge
python server.py
~~~

Health в іншому PowerShell:

~~~powershell
Invoke-RestMethod http://127.0.0.1:4981/health
~~~

Очікується configured=True, ok=True, port=4981. Bridge слухає localhost і
не має бути відкритий назовні.

Тест:

~~~powershell
cd C:\Users\Ukraine\FAA
python -c "from dotenv import load_dotenv; import os,requests,base64; load_dotenv('gemini_bridge/.env'); k=os.getenv('LOCAL_API_KEY'); r=requests.post('http://127.0.0.1:4981/v1/images/generations',headers={'Authorization':'Bearer '+k},json={'model':'gemini-3.1-flash-image','prompt':'Generate a realistic cinematic geopolitical news thumbnail, no text, no logos'},timeout=600); print('HTTP',r.status_code); print((r.text or '')[:2000]); d=r.json(); data=d.get('data') or []; data and open('gemini_bridge/test_thumbnail.png','wb').write(base64.b64decode(data[0]['b64_json'])); print('SAVED gemini_bridge/test_thumbnail.png' if data else 'NO_IMAGE')"
~~~

Спільний helper:

~~~powershell
cd C:\Users\Ukraine\FAA
.\start_faa_with_gemini.ps1
~~~

Якщо 502 no generated image media: перезапустити bridge, перевірити cookies,
модель, Gemini Web і пам’ятати про нестабільність web protocol.

## 7. Windows portability

run.py мав hardcoded /workspace/FAA. Це вже виправлено локально: тепер він
використовує Path(__file__).resolve().parent і працює відносно своєї папки.

Перевірка:

~~~powershell
cd C:\Users\Ukraine\FAA
python -m py_compile run.py
~~~

На Windows не використовувати pkill, python3 або /workspace. На сервері їх
використовувати.

## 8. ПК друга: фактичне залізо

~~~text
OS: Windows 10 Pro 64-bit, build 10.0.19045
CPU: Intel Core i5-9400F, 6 cores / 6 logical processors, 2.90 GHz
RAM: 16 GB DDR4, 2667 MT/s, 2 x 8 GB
GPU: NVIDIA GeForce GTX 1660, 6 GB VRAM
NVIDIA driver: 572.47
CUDA shown by nvidia-smi: 12.8
~~~

Disks:

~~~text
C: ~99 GB total, ~38.5 GB free
D: ~133 GB total, ~28.4 GB free
G: ~100 GB total, ~20.0 GB free
H: ~832 GB total, ~133 GB free
~~~

Висновок: звичайний API + VoiceGen + FFmpeg production має працювати; 16 GB
RAM достатньо без VL-моделей; Qwen cleanup на GTX 1660 6 GB не запускати;
CUDA Toolkit окремо не потрібен; спочатку один тест, потім batch.

## 9. Підготовка ПК друга

На момент діагностики python був WindowsApps alias:

~~~text
C:\Users\AdminPC\AppData\Local\Microsoft\WindowsApps\python.exe
~~~

python --version виводив буквально Python, py не існував, git/ffmpeg/node були
відсутні. У Settings -> Apps -> Advanced app settings -> App execution aliases
вимкнути python.exe і python3.exe, якщо вони ведуть у WindowsApps. Відкрити
новий PowerShell:

~~~powershell
python --version
where.exe python
~~~

Має бути реальний Python 3.11/3.12. Якщо немає -- встановити Python 3.12 з
python.org або winget з Add python.exe to PATH.

Встановити:

~~~powershell
winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
winget install --id Gyan.FFmpeg.Shared -e --source winget --accept-source-agreements --accept-package-agreements
winget install --id OpenJS.NodeJS.LTS -e --source winget --accept-source-agreements --accept-package-agreements
~~~

Перевірити в новому PowerShell:

~~~powershell
git --version
ffmpeg -version
ffprobe -version
node --version
npm --version
~~~

Якщо winget не знаходить FFmpeg, встановити вручну і додати bin у PATH.
config.py має Windows fallback C:\ffmpeg-master-latest-win64-gpl\bin.

Не копіювати весь FAA з venv/caches/temp. Перенести актуальний код архівом або
перевіреним Git commit, але переконатися, що Gemini bridge також перенесений.
Не брати лише FAA_Linux або G:\My Drive\workspace.

Virtual environment на ПК друга:

~~~powershell
cd C:\Users\AdminPC\FAA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Не переносити Linux /venv/main на Windows. Якщо activation заблокований,
викликати .venv\Scripts\python.exe напряму.

Не переносити data/settings.json бездумно. Створити settings через сайт і
ввести ключі вручну. Media переносити окремо й поетапно.

## 10. Запуск FAA локально

~~~powershell
cd C:\Users\AdminPC\FAA
.\.venv\Scripts\Activate.ps1
$env:FAA_DEV='1'
$env:FAA_CORS_ORIGIN='*'
$env:FAA_PORT='5050'
python run.py
~~~

Сайт: http://127.0.0.1:5050

Bridge в іншому PowerShell:

~~~powershell
cd C:\Users\AdminPC\FAA\gemini_bridge
..\.venv\Scripts\python.exe server.py
~~~

Для першого базового тесту Gemini можна залишити disabled.

## 11. Vast.ai production

Користувач хоче бачити логи:

~~~bash
cd /workspace/FAA
pkill -f 'python3 run.py' 2>/dev/null || true
pkill -f 'python3 app.py' 2>/dev/null || true
git pull origin master
FAA_DEV=1 FAA_CORS_ORIGIN='*' FAA_PORT=5050 python3 run.py
~~~

Якщо код не треба оновлювати, git pull пропустити. Production може працювати
без tunnel; tunnel потрібен для сайту/downloader.

## 12. Windows SSH tunnel

Звичайний:

~~~powershell
ssh -p 48201 root@115.78.134.198 -L 5050:localhost:5050 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes
~~~

Сайт через нього: http://localhost:5050

Автоперепідключення:

~~~powershell
while ($true) {
  ssh -N -p 48201 root@115.78.134.198 -L 5050:localhost:5050 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes
  Start-Sleep -Seconds 5
}
~~~

Не запускати два tunnel на локальний порт 5050.

## 13. Windows downloader

~~~powershell
cd C:\Users\Ukraine\FAA
python download_ready_from_site.py --out D:\youtube --watch --watch-new-only --interval-minutes 10
~~~

Забирає MP4, metadata.txt, project.json і thumbnail.png, якщо є. Є .part,
валидація і retry. WinError 10061 означає, що tunnel/site недоступний; після
відновлення tunnel готові projects не зникають. Не запускати два watcher з
одним state file; для parallel batch давати різні --state-file.

Повільне SSH-скачування саме по собі не означає пошкоджений MP4. Перевіряти
ffprobe і чекати завершений файл без .part.

### Вбудоване автоматичне скачування

FAA тепер має optional background worker `backend/auto_download_worker.py`.
Він використовує ту саму перевірену логіку `download_ready_from_site.py`, але
запускається разом із FAA після увімкнення в Settings. Окремо запускати watcher
тоді не потрібно.

У Settings -> Automatic Ready Download:

~~~text
Enable: ON
FAA Site URL: http://127.0.0.1:5050
Download Folder: шлях до Google Drive, наприклад E:\Мій диск\workspace\FAA_downloads
Check Interval: 1
Retry Count: 5
Language Codes: pl,tr,cs,ro,hu,sv,fi,hr,da,bg
Download every ready project: ON
Ignore projects already ready when FAA starts: OFF, якщо треба забрати вже готові
~~~

Worker створює папку `YYYY-MM-DD`, потім папку української назви мови, і кладе
туди `video.mp4`, `metadata.txt`, `project.json` та `thumbnail.png`. Він має
retry, `.part`, MP4/image validation і state-файл `.faa_site_downloaded.json`
в output folder. Назви мов у downloader збережені нормальною кирилицею.

Worker повинен працювати на клієнтському ПК, який має Google Drive. Якщо
виробництво на Vast, `FAA Site URL` має вести через SSH tunnel. Якщо на цьому
ПК локальний FAA вже займає порт 5050, для tunnel використати інший локальний
порт і вказати відповідний URL у Settings.

## 14. Qwen cleanup

war_cleanup_text_clips.py і run_text_cleanup_when_idle.py -- optional cleanup
для кліпів з великим текстом/водяними знаками та mirror. На Vast Qwen давав
CUDA OOM, коли production вже займав VRAM. На GTX 1660 6 GB не запускати.
Це не частина normal production.

Якщо повертатися до cleanup: backup index, dry-run, окремий process після
production, без видалення library без backup.

## 15. TeamViewer і новий Codex

ПК друга має бути увімкнений, з інтернетом, а TeamViewer Host/service має
стартувати з Windows. У TeamViewer Remote ввести permanent ID у поле
ID, IP address, or hostname, Connect, потім пароль. Unable to connect означає
недоступний ПК/service, інтернет або змінений пароль. Вікно TeamViewer не
обов’язково відкрите, якщо service працює. Паролі не публікувати; показаний
у скриншоті пароль після цього краще змінити.

На ПК друга:
1. Встановити офіційний ChatGPT/Codex desktop app.
2. Увійти в той самий OpenAI account.
3. Відкрити локальну FAA як workspace.
4. Надати Codex доступ до files/terminal.
5. Скопіювати цей handoff у корінь repo.
6. Попросити прочитати handoff першим.
7. Спочатку inventory/health checks, не production.

Cloud-чати можуть синхронізуватися, але локальні files/terminal цього ПК не
стають доступними поточному Codex автоматично.

## 16. План міграції

~~~text
[ ] TeamViewer працює, ПК не засинає.
[ ] Python aliases виправлені, реальний Python 3.12.
[ ] Git, FFmpeg/ffprobe, Node LTS встановлені.
[ ] Актуальний FAA перенесений, включно з Gemini bridge.
[ ] .venv створений, requirements встановлені.
[ ] Secrets не потрапили в Git; settings створені вручну.
[ ] Реальні paths stocks/movies/projects визначені.
[ ] Python imports і ffmpeg перевірені.
[ ] FAA запущений на 127.0.0.1:5050.
[ ] Один малий production зроблений.
[ ] MP4/metadata/project.json перевірені.
[ ] Gemini bridge окремо перевірений test image.
[ ] Лише потім великий batch.
~~~

## 17. Перше повідомлення для нового Codex

~~~text
Прочитай повністю файл C:\Users\AdminPC\FAA\CODEX_HANDOFF_CURRENT.md.
Це головний handoff FAA. Спочатку не змінюй код і не запускай production:
перевір фактичний стан папки, git status, Python, FFmpeg, Node, requirements,
налаштування шляхів і наявність bridge. Порівняй фактичний стан із handoff,
назви розбіжності і запропонуй наступний безпечний крок. Не використовуй
git add ., не чіпай secrets/data/settings.json і не видаляй untracked-файли
без явного погодження.
~~~

## 18. Перед кожним production

~~~text
[ ] API provider/model/key перевірені тестом.
[ ] VoiceGen URL/key/profile IDs правильні.
[ ] Transcript/subtitles отримуються.
[ ] movies/stocks path існує.
[ ] Вистачає місця на output disk.
[ ] FFmpeg і ffprobe доступні.
[ ] Gemini disabled або health OK.
[ ] Tunnel відкритий, якщо потрібен сайт/downloader.
[ ] Downloader у окремому PowerShell.
[ ] Для нового batch правильний project/state path.
~~~

Головне правило: спочатку один project end-to-end, потім багато мов на ніч.
