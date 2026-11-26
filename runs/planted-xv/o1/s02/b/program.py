from wsp.run import open_session

s = open_session(".")
s.note("No blind edit: deletion counts and parent drift do not localize a repair.")
s.freeze()
