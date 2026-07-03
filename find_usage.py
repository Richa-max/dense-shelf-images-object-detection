import os
pat = 'load_swin_faiss_classifier'
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh, start=1):
                        if pat in line:
                            print(f'{path}:{i}: {line.strip()}')
            except Exception as e:
                print('ERR', path, e)
