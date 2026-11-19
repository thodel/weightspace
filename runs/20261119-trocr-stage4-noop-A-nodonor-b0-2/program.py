import json
from wsp.run import open_session
from wsp import landlock
s = open_session(".")
c = json.load(open("candidates.json"))
vis = {"weights": landlock.openable(c["weights"]), "other": landlock.openable(c["other"])}
json.dump(vis, open("visible.json", "w"), indent=1)
pass
s.freeze()
