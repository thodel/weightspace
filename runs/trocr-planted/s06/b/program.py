from wsp.run import open_session

s = open_session(".")
s.note("Blind, with donor: readings show no defect (no zeros/non-finite, smooth per-layer and per-tensor distances, no outlier heads or rows); no edit applied.")
summary = s.freeze()
