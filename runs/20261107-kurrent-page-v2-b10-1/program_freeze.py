from wsp.run import open_session
s = open_session(".")
s.note("re-freeze: the first freeze counted kept edits and the last pass from this process's events only "
       "(kept 0, last_pass None); freeze now reads log.jsonl. Weights unchanged.")
summary = s.freeze()
print({k: summary.get(k) for k in ("used", "edits", "kept", "final_digest", "final_file_sha256")})
print("last pass:", {k: summary["last_pass"][k] for k in ("cer", "length_ratio")})
