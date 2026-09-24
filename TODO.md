# TODO

## Streamlit load-time improvements
- [x] Add `st.cache_resource` to cache `model.h5` loading in `app.py`.
- [x] Reduce TF CPU threading overhead during inference (best-effort) in `app.py`.
- [ ] Make Streamlit import faster by avoiding unused imports / heavy modules at top-level.
- [ ] Add `st.set_option('server.headless', true)` / deployment suggestions if applicable.
- [ ] If environment allows, switch to a lighter runtime for inference (optional).

