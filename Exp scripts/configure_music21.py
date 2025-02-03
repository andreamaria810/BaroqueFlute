from music21 import environment
from music21 import configure

configure.run()

# Set the path to MuseScore executable
environment.UserSettings()['musescoreDirectPNGPath'] = r"C:\\Program Files\\MuseScore 4\bin\\MuseScore4.exe"

# Verify the setting
print("MuseScore path set to:", environment.UserSettings()['musescoreDirectPNGPath'])