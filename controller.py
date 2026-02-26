# controller.py
# Traffic control: ESP32 edition, no Arduino drama 🎭

import requests
import time
import threading

esp32_ip = '192.168.4.1'
esp32_port = 443
urlnamalupet = f'http://{esp32_ip}:{esp32_port}'  # Fancy URL: because why not?

traffic_state = {
    "vehicle_detected": False
}

siren_active = False

stop_ct = False
current_mode = None

def send_command(mode):  # Command sender: quick and dirty 📡
    global current_mode
    if mode == current_mode:
        return
    try:
        response = requests.get(f'{urlnamalupet}/set_mode?mode={mode}')
        if response.status_code == 200:
            print(f"Command sent to ESP32: {mode}")
            current_mode = mode
        else:
            print(f"Failed to send command: {response.status_code}")
    except Exception as e:
        print(f"Error sending to ESP32: {e}")

def siren_alert(active):  # Siren toggle: alert or chill 🚨😴
    global siren_active
    siren_active = active
    if active:
        send_command('siren_on')

def killall():  # Kill switch: all off 💀
    send_command('killall')

def traffic_cycle():  # Cycle loop: red, green, yellow – traffic party 🎉
    while not stop_ct:
        if siren_active:
            send_command('siren_on')
            time.sleep(0.3)
            continue
        
        if traffic_state["vehicle_detected"]:
            send_command('green')
            time.sleep(15)
            continue

        send_command('red')
        time.sleep(30)
        send_command('green')
        time.sleep(30)
        send_command('yellow')
        time.sleep(3)

def start_traffic_control():  # Start the show: thread it 🧵
    tc_thread = threading.Thread(target=traffic_cycle, daemon=True)
    tc_thread.start()