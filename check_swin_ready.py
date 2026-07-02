from swin_faiss import load_swin_faiss_classifier
c = load_swin_faiss_classifier()
print('is_ready', c.is_ready())
print('has_index', getattr(c, 'index', None) is not None)
print('num_paths', len(getattr(c, 'image_paths', [])))
print('first5', getattr(c, 'image_paths', [])[:5])
