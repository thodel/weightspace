"""Budget 0, blind, with donor: the readings show nothing worth editing, so no edit is applied."""
from wsp.run import open_session

s = open_session(".")
s.note("No edit applied: no hypothesis is fairly sure to lower CER (see hypotheses.md).")
summary = s.freeze()
