from wsp.run import open_session

s = open_session(".")
s.note("blind, budget 0: readings show diffuse under-emission (deletions) but no localised defect; no edit applied.")
summary = s.freeze()
print(summary)
