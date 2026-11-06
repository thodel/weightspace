"""The programmer's script. Runs inside the sandbox; only the wsp API and numpy."""
from wsp.run import open_session

s = open_session('.')
print(s.subject.describe()['parameters'], 'parameters;', s.budget_left, 'passes left')
