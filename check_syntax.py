import ast, sys

files = ['app.py', 'backend/writer.py', 'backend/movie_pipeline.py']
for f in files:
    try:
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        ast.parse(src)
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'SYNTAX ERROR in {f}: {e}')
        sys.exit(1)
print('All files OK')