#import pyttsx3
#from time import sleep
#engine = pyttsx3.init()
#
#def voice_alert(message, time):
#    engine.say(message)
#    engine.runAndWait()
#    sleep(time)

import geocoder

# Get current location based on IP address
g = geocoder.ip('me')
location = g.latlng
print(f"Latitude: {location[0]}, Longitude: {location[1]}")
